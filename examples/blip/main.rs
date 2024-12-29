use usls::{models::Blip, DataLoader, Options};

#[derive(argh::FromArgs)]
/// BLIP Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image
    #[argh(option, default = "vec![String::from(\"./assets/bus.jpg\")]")]
    source: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();

    // build model
    let options_visual = Options::blip_v1_base_caption_visual()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let options_textual = Options::blip_v1_base_caption_textual()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = Blip::new(options_visual, options_textual)?;

    // image caption
    let xs = DataLoader::try_read_batch(&args.source)?;

    // unconditional caption
    let ys = model.forward(&xs, None)?;
    println!("Unconditional: {:?}", ys);

    // conditional caption
    let ys = model.forward(&xs, Some("this image depict"))?;
    println!("Conditional: {:?}", ys);

    Ok(())
}
