use usls::{models::YOLO, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default().with_model("yolov8-falldown-f16.onnx")?;
    let mut model = YOLO::new(&options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/falldown.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("YOLOv8-Falldown");
    annotator.annotate(&x, &y);

    Ok(())
}
