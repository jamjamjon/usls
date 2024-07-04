use usls::{coco, models::YOLO, Annotator, DataLoader, Options, Vision};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        // .with_model("yolov8m-dyn.onnx")?
        // .with_model("yolov8m-dyn-f16.onnx")?
        // .with_model("yolov8m-pose-dyn.onnx")?
        // .with_model("yolov8m-cls-dyn.onnx")?
        .with_model("yolov8m-seg-dyn.onnx")?
        // .with_model("yolov8m-obb-dyn.onnx")?
        // .with_model("yolov8m-oiv7-dyn.onnx")?
        // .with_trt(0)
        // .with_fp16(true)
        // .with_coreml(0)
        // .with_cuda(3)
        .with_i00((1, 1, 4).into())
        .with_i02((224, 640, 800).into())
        .with_i03((224, 640, 800).into())
        .with_confs(&[0.4, 0.15]) // class_0: 0.4, others: 0.15
        .with_names2(&coco::KEYPOINTS_NAMES_17)
        .with_profile(false);
    let mut model = YOLO::new(options)?;

    // build dataloader
    let dl = DataLoader::default()
        .with_batch(model.batch() as _)
        .load("./assets/bus.jpg")?;

    // build annotator
    let annotator = Annotator::default()
        .with_skeletons(&coco::SKELETONS_16)
        .with_bboxes_thickness(7)
        .with_saveout("YOLOv8");

    // run & annotate
    for (xs, _paths) in dl {
        // let ys = model.run(&xs)?;  // way one
        let ys = model.forward(&xs, true)?; // way two
        annotator.annotate(&xs, &ys);
    }

    Ok(())
}
