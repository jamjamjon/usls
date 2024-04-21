use usls::{models::YOLO, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default().with_model("../models/yolov8-head-f16.onnx");
    let mut model = YOLO::new(&options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/kids.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("YOLOv8-Head");
    annotator.annotate(&x, &y);

    Ok(())
}
