use anyhow::Result;
use usls::{models::Ram, Config, DataLoader};

/// RAM & RAM++ Example
#[derive(argh::FromArgs)]
struct Args {
    /// source image
    #[argh(
        option,
        default = "vec![
           String::from(\"./assets/bus.jpg\"),
            String::from(\"./assets/dog.jpg\"),
            String::from(\"./assets/cat.png\")
        ]"
    )]
    source: Vec<String>,

    /// device
    #[argh(option, default = "String::from(\"cpu\")")]
    device: String,

    /// dtype
    #[argh(option, default = "String::from(\"bnb4\")")]
    dtype: String,

    /// kind
    #[argh(option, default = "String::from(\"ram\")")]
    kind: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // load model
    let config = match args.kind.as_str() {
        "ram" => Config::ram(),
        "ram++" => Config::ram_plus(),
        _ => unimplemented!("Unsupported Recognize Anything Model kind: {:?}", args.kind),
    }
    .with_model_device(args.device.parse()?)
    .with_model_dtype(args.dtype.parse()?)
    .with_batch_size_all(args.source.len())
    .commit()?;
    let mut model = Ram::new(config)?;

    // load images
    let xs = DataLoader::try_read_n(&args.source)?;

    // batch forward
    let ys = model.forward(&xs)?;

    // results
    for (x, y) in xs.iter().zip(ys.iter()) {
        if let Some(texts) = y.texts() {
            println!("Image: {:?}", x);
            println!("Texts: {:?}", texts);
        }
        println!("--------------------------------");
    }

    // summary
    usls::perf(false);

    Ok(())
}
