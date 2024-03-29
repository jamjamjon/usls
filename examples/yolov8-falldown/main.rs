use usls::{models::YOLO, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/yolov8-falldown-f16.onnx")
        .with_confs(&[0.3])
        .with_saveout("YOLOv8-Falldown")
        .with_profile(false);
    let mut model = YOLO::new(&options)?;

    // build dataloader
    let mut dl = DataLoader::default().load("./assets/falldown.jpg")?;

    // run
    model.run(&dl.next().unwrap().0)?;

    Ok(())
}
