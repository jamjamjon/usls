use anyhow::Result;
use usls::{models::GroundingDINO, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
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

    let options = Options::grounding_dino_tiny()
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .with_model_device(args.device.as_str().try_into()?)
        .with_text_names(&args.labels.iter().map(|x| x.as_str()).collect::<Vec<_>>())
        .with_class_confs(&[0.25])
        .with_text_confs(&[0.25])
        .commit()?;

    let mut model = GroundingDINO::new(options)?;

    // load images
    let xs = DataLoader::try_read_batch(&args.source)?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default()
        .with_bboxes_thickness(4)
        .with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    // summary
    model.summary();

    Ok(())
}
