use anyhow::Result;
use usls::{
    models::{SamPrompt, SAM2, YOLO},
    Annotator, DataLoader, Options, Scale, Style,
};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build SAM
    let (options_encoder, options_decoder) = (
        Options::sam2_1_tiny_encoder().commit()?,
        Options::sam2_1_tiny_decoder().commit()?,
    );
    let mut sam = SAM2::new(options_encoder, options_decoder)?;

    // build YOLOv8
    let options_yolo = Options::yolo_detect()
        .with_model_scale(Scale::N)
        .with_model_version(8.into())
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut yolo = YOLO::new(options_yolo)?;

    // load one image
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

    // build annotator
    let annotator = Annotator::default()
        .with_polygon_style(
            Style::polygon()
                .with_visible(true)
                .with_text_visible(true)
                .show_id(true)
                .show_name(true),
        )
        .with_mask_style(Style::mask().with_draw_mask_polygon_largest(true));

    // run & annotate
    let ys_det = yolo.forward(&xs)?;
    for y_det in ys_det.iter() {
        if let Some(hbbs) = y_det.hbbs() {
            // collect hhbs
            let mut prompt = SamPrompt::default();
            for hbb in hbbs {
                prompt = prompt.with_xyxy(hbb.xmin(), hbb.ymin(), hbb.xmax(), hbb.ymax());
            }

            // sam2 infer
            let ys_sam = sam.forward(&xs, &[prompt])?;

            // annotate
            for (x, y) in xs.iter().zip(ys_sam.iter()) {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs", "YOLO-SAM2"])?
                        .join(usls::timestamp(None))
                        .display(),
                ))?;
            }
        }
    }

    Ok(())
}
