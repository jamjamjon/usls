use usls::{models::ImageClassifier, Annotator, DataLoader, Options};

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
    #[argh(
        option,
        default = "vec![
            String::from(\"images/dog.jpg\"),
            String::from(\"images/siamese.png\"),
            String::from(\"images/ailurus-fulgens.jpg\"),
        ]"
    )]
    source: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();

    // build model
    let options = Options::beit_base()
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = ImageClassifier::try_from(options)?;

    // load images
    let xs = DataLoader::try_read_batch(&args.source)?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default().with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    Ok(())
}
