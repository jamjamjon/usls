use usls::{models::YOLO, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/yolov8-head-f16.onnx")
        .with_confs(&[0.3])
        .with_saveout("YOLOv8-Head")
        .with_profile(false);
    let mut model = YOLO::new(&options)?;

    // build dataloader
    let mut dl = DataLoader::default().load("./assets/kids.jpg")?;

    // run
    model.run(&dl.next().unwrap().0)?;

    Ok(())
}
