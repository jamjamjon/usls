use usls::{coco, models::RTDETR, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("rtdetr-l-f16.onnx")?
        .with_confs(&[0.4, 0.15])
        .with_names(&coco::NAMES_80);
    let mut model = RTDETR::new(options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/bus.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("RT-DETR");
    annotator.annotate(&x, &y);

    Ok(())
}
