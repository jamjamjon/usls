use anyhow::Result;
use usls::{models::YOLO, Annotator, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build model
    let config = Config::doclayout_yolo_docstructbench()
        .with_model_device(args.device.parse()?)
        .commit()?;
    let mut model = YOLO::new(config)?;

    // load images
    let xs = DataLoader::try_read_n(&["images/academic.jpg"])?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default();
    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", "doclayout-yolo"])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }
    usls::perf(false);

    Ok(())
}
