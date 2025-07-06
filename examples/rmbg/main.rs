use usls::{models::RMBG, Annotator, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// version
    #[argh(option, default = "1.4")]
    ver: f32,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    let config = match args.ver {
        1.4 => Config::rmbg1_4(),
        2.0 => Config::rmbg2_0(),
        _ => unreachable!("Unsupported version"),
    };

    // build model
    let config = config
        .with_model_dtype(args.dtype.parse()?)
        .with_model_device(args.device.parse()?)
        .commit()?;
    let mut model = RMBG::new(config)?;

    // load image
    let xs = DataLoader::try_read_n(&["./assets/cat.png"])?;

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
