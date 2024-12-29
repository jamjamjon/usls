use anyhow::Result;
use usls::{models::Sapiens, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();
    // build
    let options = Options::sapiens_seg_0_3b()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = Sapiens::new(options)?;

    // load
    let x = [DataLoader::try_read("images/paul-george.jpg")?];

    // run
    let y = model.forward(&x)?;

    // annotate
    let annotator = Annotator::default()
        .without_masks(true)
        .with_polygons_name(true)
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    Ok(())
}
