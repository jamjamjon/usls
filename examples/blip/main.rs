use usls::{models::Blip, Config, DataLoader};

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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = Config::blip_v1_base_caption()
        .with_device_all(args.device.parse()?)
        .commit()?;
    let mut model = Blip::new(config)?;

    // image caption
    let xs = DataLoader::try_read_n(&args.source)?;

    // unconditional caption
    let ys = model.forward(&xs, None)?;
    println!("Unconditional: {:?}", ys);

    // conditional caption
    let ys = model.forward(&xs, Some("this image depict"))?;
    println!("Conditional: {:?}", ys);

    usls::perf(false);

    Ok(())
}
