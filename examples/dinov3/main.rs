use anyhow::Result;
use usls::{models::DINOv3, Config, DataLoader};

/// DINOv3 Example
#[derive(argh::FromArgs)]
struct Args {
    /// cpu:0, cuda:0
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// fp32, fp16, q8, q4, q4f16, bnb4
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // images
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg", "./assets/bus.jpg"])?;

    // model
    let config = Config::dinov3_vits16_lvd1689m()
        .with_batch_size_all(xs.len())
        .with_dtype_all(args.dtype.parse()?)
        .with_device_all(args.device.parse()?)
        .commit()?;
    let mut model = DINOv3::new(config)?;

    // encode images
    let feats = model.encode_images(&xs)?;
    println!("Feat shape: {:?}", feats.shape());

    usls::perf(false);

    Ok(())
}
