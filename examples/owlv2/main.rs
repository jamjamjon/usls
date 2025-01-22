use anyhow::Result;
use usls::{models::OWLv2, Annotator, DataLoader, Options};

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
            String::from(\"hand\"), 
            String::from(\"shoes\"), 
            String::from(\"bus\"), 
            String::from(\"car\"), 
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

    // options
    let options = Options::owlv2_base_ensemble()
        // owlv2_base()
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .with_model_device(args.device.as_str().try_into()?)
        .with_class_names(&args.labels.iter().map(|x| x.as_str()).collect::<Vec<_>>())
        .commit()?;
    let mut model = OWLv2::new(options)?;

    // load
    let xs = DataLoader::try_read_batch(&args.source)?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    Ok(())
}
