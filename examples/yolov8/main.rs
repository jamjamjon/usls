use usls::{models::YOLO, DataLoader, Options, COCO_SKELETON_17};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1.build model
    let options = Options::default()
        .with_model("../models/yolov8m-dyn-f16.onnx")
        .with_trt(0) // cuda by default
        .with_fp16(true)
        .with_i00((1, 1, 4).into())
        .with_i02((416, 640, 800).into())
        .with_i03((416, 640, 800).into())
        .with_confs(&[0.4, 0.15]) // person: 0.4, others: 0.15
        .with_profile(true)
        .with_dry_run(5)
        .with_skeletons(&COCO_SKELETON_17)
        .with_saveout("YOLOv8");
    let mut model = YOLO::new(&options)?;

    // 2.build dataloader
    let dl = DataLoader::default()
        .with_batch(1)
        .load("./assets/bus.jpg")?;

    // 3.run
    for (xs, _paths) in dl {
        let _y = model.run(&xs)?;
    }
    Ok(())
}
