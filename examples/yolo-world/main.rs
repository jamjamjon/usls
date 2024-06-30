use usls::{models::YOLO, Annotator, DataLoader, Options, Vision};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("yolov8s-world-v2-shoes.onnx")?
        .with_i00((1, 1, 4).into())
        .with_i02((416, 640, 800).into())
        .with_i03((416, 640, 800).into())
        .with_confs(&[0.3])
        .with_profile(false);
    let mut model = YOLO::new(options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/bus.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("YOLO-World");
    annotator.annotate(&x, &y);

    Ok(())
}
