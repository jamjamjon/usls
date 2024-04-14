use usls::{models::YOLO, Annotator, DataLoader, Options, COCO_SKELETON_17};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        // .with_model("../models/yolov8m-seg-dyn-f16.onnx")
        .with_model("../models/yolov8m-cls.onnx")
        // .with_trt(0) // cuda by default
        // .with_fp16(true)
        .with_i00((1, 1, 4).into())
        .with_i02((224, 224, 800).into())
        .with_i03((224, 224, 800).into())
        .with_confs(&[0.4, 0.15]) // person: 0.4, others: 0.15
        .with_profile(false)
        .with_dry_run(3);
    let mut model = YOLO::new(&options)?;

    // build dataloader
    let dl = DataLoader::default()
        .with_batch(1)
        .load("./assets/bus.jpg")?;

    // build annotate
    let annotator = Annotator::default()
        .with_skeletons(&COCO_SKELETON_17)
        .without_conf(false)
        .without_name(false)
        .without_masks(false)
        .without_polygons(false)
        .without_bboxes(false)
        .with_saveout("YOLOv8");

    // run & annotate
    for (xs, _paths) in dl {
        let ys = model.run(&xs)?;
        println!("{:?}", ys);
        annotator.annotate(&xs, &ys);
    }

    Ok(())
}
