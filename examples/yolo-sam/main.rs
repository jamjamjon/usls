use anyhow::Result;
use usls::{
    models::{SamPrompt, SAM, YOLO},
    Annotator, DataLoader, Options, Scale,
};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    // tracing_subscriber::fmt()
    //     .with_max_level(tracing::Level::INFO)
    //     .init();

    let args: Args = argh::from_env();

    // build SAM
    let (options_encoder, options_decoder) = (
        Options::mobile_sam_tiny_encoder().commit()?,
        Options::mobile_sam_tiny_decoder().commit()?,
    );
    let mut sam = SAM::new(options_encoder, options_decoder)?;

    // build YOLOv8
    let options_yolo = Options::yolo_detect()
        .with_model_scale(Scale::N)
        .with_model_version(8.0.into())
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut yolo = YOLO::new(options_yolo)?;

    // load one image
    let xs = DataLoader::try_read_batch(&["images/dog.jpg"])?;

    // build annotator
    let annotator = Annotator::default()
        .with_bboxes_thickness(7)
        .without_bboxes_name(true)
        .without_bboxes_conf(true)
        .without_mbrs(true)
        .with_saveout("YOLO-SAM");

    // run & annotate
    let ys_det = yolo.forward(&xs)?;
    for y_det in ys_det.iter() {
        if let Some(bboxes) = y_det.bboxes() {
            for bbox in bboxes {
                let ys_sam = sam.forward(
                    &xs,
                    &[SamPrompt::default().with_bbox(
                        bbox.xmin(),
                        bbox.ymin(),
                        bbox.xmax(),
                        bbox.ymax(),
                    )],
                )?;
                annotator.annotate(&xs, &ys_sam);
            }
        }
    }

    Ok(())
}
