use anyhow::Result;
use usls::{models::RTDETR, Annotator, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// model file
    #[argh(option)]
    model: Option<String>,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

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

    // config
    let config = match &args.model {
        Some(m) => Config::deimv2().with_model_file(m),
        None => Config::deim_v2_n_coco(),
    }
    .with_model_dtype(args.dtype.parse()?)
    .with_device_all(args.device.parse()?)
    .commit()?;
    let mut model = RTDETR::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default();
    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", model.spec()])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    usls::perf(false);

    Ok(())
}
