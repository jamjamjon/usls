use anyhow::Result;
use usls::{models::RTMO, Annotator, DataLoader, Options, COCO_SKELETONS_16};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // build model
    let mut model = RTMO::new(Options::rtmo_s().commit()?)?;

    // load image
    let xs = [DataLoader::try_read("images/bus.jpg")?];

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default()
        .with_saveout(model.spec())
        .with_skeletons(&COCO_SKELETONS_16);
    annotator.annotate(&xs, &ys);

    Ok(())
}
