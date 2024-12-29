use anyhow::Result;
use usls::{models::DepthPro, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// BLIP Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype
    #[argh(option, default = "String::from(\"q4f16\")")]
    dtype: String,

    /// source image
    #[argh(option, default = "String::from(\"images/street.jpg\")")]
    source: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Args = argh::from_env();

    // model
    let options = Options::depth_pro()
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = DepthPro::new(options)?;

    // load
    let x = [DataLoader::try_read(&args.source)?];

    // run
    let y = model.forward(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_colormap("Turbo")
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    Ok(())
}
