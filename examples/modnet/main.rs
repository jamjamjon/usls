use usls::{models::MODNet, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("modnet/dyn-f32.onnx")?
        .with_ixx(0, 2, (416, 512, 800).into())
        .with_ixx(0, 3, (416, 512, 800).into());
    let mut model = MODNet::new(options)?;

    // load image
    let x = [DataLoader::try_read("images/liuyifei.png")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default().with_saveout("MODNet");
    annotator.annotate(&x, &y);

    Ok(())
}
