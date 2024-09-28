use usls::{
    models::{SamKind, SamPrompt, YOLOTask, YOLOVersion, SAM, YOLO},
    Annotator, DataLoader, Options, Vision,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build SAM
    let options_encoder = Options::default().with_model("sam/mobile-sam-vit-t-encoder.onnx")?;
    let options_decoder = Options::default()
        .with_find_contours(true)
        .with_sam_kind(SamKind::Sam)
        .with_model("sam/mobile-sam-vit-t-decoder.onnx")?;
    let mut sam = SAM::new(options_encoder, options_decoder)?;

    // build YOLOv8-Det
    let options_yolo = Options::default()
        .with_yolo_version(YOLOVersion::V8)
        .with_yolo_task(YOLOTask::Detect)
        .with_model("yolo/v8-m-dyn.onnx")?
        .with_cuda(0)
        .with_ixx(0, 2, (416, 640, 800).into())
        .with_ixx(0, 3, (416, 640, 800).into())
        .with_find_contours(false)
        .with_confs(&[0.45]);
    let mut yolo = YOLO::new(options_yolo)?;

    // load one image
    let xs = [DataLoader::try_read("images/dog.jpg")?];

    // build annotator
    let annotator = Annotator::default()
        .with_bboxes_thickness(7)
        .without_bboxes_name(true)
        .without_bboxes_conf(true)
        .without_mbrs(true)
        .with_saveout("YOLO-SAM");

    // run & annotate
    let ys_det = yolo.run(&xs)?;
    for y_det in ys_det {
        if let Some(bboxes) = y_det.bboxes() {
            for bbox in bboxes {
                let ys_sam = sam.run(
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
