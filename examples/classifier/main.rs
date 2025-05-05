use usls::{models::ImageClassifier, Annotator, DataLoader, ImageClassificationConfig, Options};

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
    #[argh(option, default = "String::from(\"beit\")")]
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
        "beit" => ImageClassificationConfig::beit_base(),
        "convnext" => ImageClassificationConfig::convnext_v2_atto(),
        "deit" => ImageClassificationConfig::deit_tiny_distill(),
        "fastvit" => ImageClassificationConfig::fastvit_t8_distill(),
        "mobileone" => ImageClassificationConfig::mobileone_s0(),
        _ => anyhow::bail!("Unsupported model: {}", args.model),
    }
    .with_model_dtype(args.dtype.as_str().try_into()?)
    .with_model_device(args.device.as_str().try_into()?);

    // let options = options
    //     .with_model_dtype(args.dtype.as_str().try_into()?)
    //     .with_model_device(args.device.as_str().try_into()?)
    //     .commit()?;
    let mut model = ImageClassifier::try_from(config)?;
    // let config = match &args.model {
    //     Some(m) => DBConfig::default().with_model_file(m),
    //     None => DBConfig::ppocr_det_v4_ch().with_model_dtype(args.dtype.as_str().try_into()?),
    // }
    // .with_model_device(args.device.as_str().try_into()?);
    // let mut model = DB::new(config)?;

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

    Ok(())
}
