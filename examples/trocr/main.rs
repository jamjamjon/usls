use usls::{
    models::{TrOCR, TrOCRKind},
    Config, DataLoader, Scale,
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
    let config = match args.scale.parse()? {
        Scale::S => match args.kind.parse()? {
            TrOCRKind::Printed => Config::trocr_small_printed(),
            TrOCRKind::HandWritten => Config::trocr_small_handwritten(),
        },
        Scale::B => match args.kind.parse()? {
            TrOCRKind::Printed => Config::trocr_base_printed(),
            TrOCRKind::HandWritten => Config::trocr_base_handwritten(),
        },
        x => anyhow::bail!("Unsupported TrOCR scale: {:?}", x),
    }
    .with_device_all(args.device.parse()?)
    .with_dtype_all(args.dtype.parse()?)
    .commit()?;

    let mut model = TrOCR::new(config)?;

    // inference
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    usls::perf(false);

    Ok(())
}
