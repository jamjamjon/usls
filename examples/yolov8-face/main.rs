use usls::{models::YOLO, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/yolov8n-face-dyn-f16.onnx")
        .with_i00((1, 1, 4).into())
        .with_i02((416, 640, 800).into())
        .with_i03((416, 640, 800).into())
        .with_confs(&[0.15])
        .with_saveout("YOLOv8-Face")
        .with_profile(false);
    let mut model = YOLO::new(&options)?;

    // build dataloader
    let mut dl = DataLoader::default().load("./assets/kids.jpg")?;

    // run
    model.run(&dl.next().unwrap().0)?;

    Ok(())
}
