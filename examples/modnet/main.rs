use usls::{models::MODNet, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("modnet-dyn.onnx")?
        .with_i00((1, 1, 4).into())
        .with_i02((416, 512, 800).into())
        .with_i03((416, 512, 800).into());
    let mut model = MODNet::new(&options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/portrait.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("MODNet");
    annotator.annotate(&x, &y);

    Ok(())
}
