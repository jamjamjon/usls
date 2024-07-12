use usls::{models::DepthAnything, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // options
    let options = Options::default()
        // .with_model("depth-anything-s-dyn.onnx")?
        .with_model("depth-anything-v2-s.onnx")?
        .with_i00((1, 1, 8).into())
        .with_i02((384, 512, 1024).into())
        .with_i03((384, 512, 1024).into());
    let mut model = DepthAnything::new(options)?;

    // load
    let x = vec![DataLoader::try_read("./assets/2.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_colormap("Turbo")
        .with_saveout("Depth-Anything");
    annotator.annotate(&x, &y);

    Ok(())
}
