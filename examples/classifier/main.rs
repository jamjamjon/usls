use usls::{models::ImageClassifier, Annotator, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// model name
    #[argh(option, default = "String::from(\"mobileone\")")]
    model: String,

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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build model
    let config = match args.model.to_lowercase().as_str() {
        "beit" => Config::beit_base(),
        "convnext" => Config::convnext_v2_atto(),
        "deit" => Config::deit_tiny_distill(),
        "fastvit" => Config::fastvit_t8_distill(),
        "mobileone" => Config::mobileone_s0(),
        _ => anyhow::bail!("Unsupported model: {}", args.model),
    };

    let config = config
        .with_model_dtype(args.dtype.parse()?)
        .with_model_device(args.device.parse()?)
        .commit()?;
    let mut model = ImageClassifier::try_from(config)?;

    // load images
    let xs = DataLoader::try_read_n(&args.source)?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default();
    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", "Image-Classifier", &args.model])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    usls::perf(false);

    Ok(())
}
