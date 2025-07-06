use anyhow::Result;
use usls::{models::DepthAnything, Annotator, Config, DataLoader, Style};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // build model
    let mut model = DepthAnything::new(Config::depth_anything_v2_small().commit()?)?;

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
