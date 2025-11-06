use anyhow::Result;
use usls::{models::FastVLM, Config, DataLoader, Scale};

/// Example
#[derive(argh::FromArgs)]
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image
    #[argh(option, default = "vec![String::from(\"./assets/bus.jpg\")]")]
    source: Vec<String>,

    /// promt
    #[argh(option, default = "String::from(\"Describe the image in detail.\")")]
    prompt: String,

    /// scale
    #[argh(option, default = "String::from(\"0.5b\")")]
    scale: String,

    /// max_tokens
    #[argh(option, default = "1024")]
    max_tokens: usize,

    /// dtype
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = match args.scale.parse()? {
        Scale::Billion(0.5) => Config::fastvlm_0_5b(),
        _ => unimplemented!(),
    }
    .with_device_all(args.device.parse()?)
    .with_dtype_all(args.dtype.parse()?)
    .with_batch_size_all(args.source.len())
    .with_max_tokens(args.max_tokens)
    .commit()?;
    let mut model = FastVLM::new(config)?;

    // load images
    let xs = DataLoader::try_read_n(&args.source)?;

    // run
    let ys = model.forward(&xs, &args.prompt)?;

    for y in ys.iter() {
        let texts = y.texts();
        if !texts.is_empty() {
            for text in texts {
                println!("\n[User]: {}\n[Assistant]:{:?}", args.prompt, text);
            }
        }
    }
    usls::perf(false);

    Ok(())
}
