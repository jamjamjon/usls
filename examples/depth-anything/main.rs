use usls::{models::DepthAnything, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // visual
    let options = Options::default()
        .with_model("../models/depth-anything-s-dyn.onnx")?
        .with_i00((1, 1, 8).into())
        .with_i02((384, 512, 1024).into())
        .with_i03((384, 512, 1024).into());
    let model = DepthAnything::new(&options)?;

    // load
    let x = vec![DataLoader::try_read("./assets/2.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_colormap_turbo(false)
        .with_colormap_inferno(true)
        .with_saveout("Depth-Anything");
    annotator.annotate(&x, &y);

    Ok(())
}
