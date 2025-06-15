use anyhow::Result;
use usls::DataLoader;
use usls::{models::DepthPro, Annotator, Config, Style};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype
    #[argh(option, default = "String::from(\"q4f16\")")]
    dtype: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // model
    let config = Config::depth_pro()
        .with_model_dtype(args.dtype.parse()?)
        .with_model_device(args.device.parse()?)
        .commit()?;

    let mut model = DepthPro::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&["images/street.jpg"])?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator =
        Annotator::default().with_mask_style(Style::mask().with_colormap256("turbo".parse()?));
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
