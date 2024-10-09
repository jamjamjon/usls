use usls::{models::DepthPro, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // options
    let options = Options::default()
        .with_model("depth-pro/q4f16.onnx")? // bnb4, f16
        .with_ixx(0, 0, 1.into()) // batch. Note: now only support batch_size = 1
        .with_ixx(0, 1, 3.into()) // channel
        .with_ixx(0, 2, 1536.into()) // height
        .with_ixx(0, 3, 1536.into()); // width
    let mut model = DepthPro::new(options)?;

    // load
    let x = [DataLoader::try_read("images/street.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_colormap("Turbo")
        .with_saveout("Depth-Pro");
    annotator.annotate(&x, &y);

    Ok(())
}
