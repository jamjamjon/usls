use usls::{models::MediaPipeSegmenter, Annotator, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = Config::mediapipe_selfie_segmentater()
        .with_model_dtype(args.dtype.parse()?)
        .with_model_device(args.device.parse()?)
        .commit()?;
    let mut model = MediaPipeSegmenter::new(config)?;

    // load image
    let xs = DataLoader::try_read_n(&["images/selfie-segmenter.png"])?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator =
        Annotator::default().with_mask_style(usls::Style::mask().with_mask_cutout(true));
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
