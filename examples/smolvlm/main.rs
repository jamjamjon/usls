use anyhow::Result;
use usls::{models::SmolVLM, Config, DataLoader, Scale};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image
    #[argh(option, default = "vec![String::from(\"./assets/bus.jpg\")]")]
    source: Vec<String>,

    /// promt
    #[argh(option, default = "String::from(\"Can you describe this image?\")")]
    prompt: String,

    /// scale
    #[argh(option, default = "String::from(\"256m\")")]
    scale: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = match args.scale.parse()? {
        Scale::Million(256.) => Config::smolvlm_256m(),
        Scale::Million(500.) => Config::smolvlm_500m(),
        _ => unimplemented!(),
    }
    .with_device_all(args.device.parse()?)
    .commit()?;
    let mut model = SmolVLM::new(config)?;

    // load images
    let xs = DataLoader::try_read_n(&args.source)?;

    // run
    let ys = model.forward(&xs, &args.prompt)?;

    for y in ys.iter() {
        if let Some(texts) = y.texts() {
            for text in texts {
                println!("[User]: {}\n\n[Assistant]:{:?}", args.prompt, text);
            }
        }
    }
    usls::perf(false);

    Ok(())
}
