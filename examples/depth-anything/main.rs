use usls::{models::DepthAnything, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // options
    let options = Options::default()
        // .with_model("depth-anything/v1-s-dyn.onnx")?
        .with_model("depth-anything/v2-s.onnx")?
        .with_ixx(0, 2, (384, 512, 1024).into())
        .with_ixx(0, 3, (384, 512, 1024).into());
    let mut model = DepthAnything::new(options)?;

    // load
    let x = [DataLoader::try_read("images/street.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_colormap("Turbo")
        .with_saveout("Depth-Anything");
    annotator.annotate(&x, &y);

    Ok(())
}
