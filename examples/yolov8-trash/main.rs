use usls::{models::YOLO, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1.build model
    let options = Options::default()
        .with_model("yolov8-plastic-bag-f16.onnx")?
        .with_names(&["trash"]);
    let mut model = YOLO::new(&options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/trash.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("YOLOv8-Trash");
    annotator.annotate(&x, &y);

    Ok(())
}
