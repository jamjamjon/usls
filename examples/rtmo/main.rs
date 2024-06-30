use usls::{coco, models::RTMO, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("rtmo-s-dyn.onnx")?
        .with_i00((1, 1, 8).into())
        .with_nk(17)
        .with_confs(&[0.3])
        .with_kconfs(&[0.5]);
    let mut model = RTMO::new(options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/bus.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_saveout("RTMO")
        .with_skeletons(&coco::SKELETONS_16);
    annotator.annotate(&x, &y);

    Ok(())
}
