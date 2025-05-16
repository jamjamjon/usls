use usls::{
    models::{TrOCR, TrOCRKind},
    DataLoader, ModelConfig, Scale,
};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale
    #[argh(option, default = "String::from(\"s\")")]
    scale: String,

    /// kind
    #[argh(option, default = "String::from(\"printed\")")]
    kind: String,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // load images
    let xs = DataLoader::try_read_n(&[
        "images/text-en-dark.png",
        "images/text-hello-rust-handwritten.png",
    ])?;

    // build model
    let config = match args.scale.as_str().try_into()? {
        Scale::S => match args.kind.as_str().try_into()? {
            TrOCRKind::Printed => ModelConfig::trocr_small_printed(),
            TrOCRKind::HandWritten => ModelConfig::trocr_small_handwritten(),
        },
        Scale::B => match args.kind.as_str().try_into()? {
            TrOCRKind::Printed => ModelConfig::trocr_base_printed(),
            TrOCRKind::HandWritten => ModelConfig::trocr_base_handwritten(),
        },
        x => anyhow::bail!("Unsupported TrOCR scale: {:?}", x),
    }
    .with_device_all(args.device.as_str().try_into()?)
    .with_dtype_all(args.dtype.as_str().try_into()?)
    .commit()?;

    let mut model = TrOCR::new(config)?;

    // inference
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // summary
    model.summary();

    Ok(())
}
