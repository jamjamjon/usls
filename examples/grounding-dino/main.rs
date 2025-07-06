use anyhow::Result;
use usls::{models::GroundingDINO, Annotator, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"q8\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image
    #[argh(option, default = "vec![String::from(\"./assets/bus.jpg\")]")]
    source: Vec<String>,

    /// open class names
    #[argh(
        option,
        default = "vec![
            String::from(\"person\"),
            String::from(\"a hand\"),
            String::from(\"a shoe\"),
            String::from(\"bus\"),
            String::from(\"dog\"),
            String::from(\"cat\"),
            String::from(\"sign\"),
            String::from(\"tie\"),
            String::from(\"monitor\"),
            String::from(\"glasses\"),
            String::from(\"tree\"),
            String::from(\"head\"),
        ]"
    )]
    labels: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    let config = Config::grounding_dino_tiny()
        .with_model_dtype(args.dtype.parse()?)
        .with_model_device(args.device.parse()?)
        .with_text_names(&args.labels.iter().map(|x| x.as_str()).collect::<Vec<_>>())
        .with_class_confs(&[0.25])
        .with_text_confs(&[0.25])
        .commit()?;

    let mut model = GroundingDINO::new(config)?;

    // load images
    let xs = DataLoader::try_read_n(&args.source)?;

    // run
    let ys = model.forward(&xs)?;

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

    // summary
    usls::perf(false);

    Ok(())
}
