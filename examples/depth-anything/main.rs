use anyhow::Result;
use usls::{models::DepthAnything, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // build model
    let options = Options::depth_anything_v2_small().commit()?;
    let mut model = DepthAnything::new(options)?;

    // load
    let x = [DataLoader::try_read("images/street.jpg")?];

    // run
    let y = model.forward(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_colormap("Turbo")
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    Ok(())
}
