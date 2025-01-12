use anyhow::Result;
use usls::{models::YOLOPv2, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // build model
    let options = Options::yolop_v2_480x800().commit()?;
    let mut model = YOLOPv2::new(options)?;

    // load image
    let x = [DataLoader::try_read("images/car-view.jpg")?];

    // run
    let y = model.forward(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_polygons_name(true)
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    Ok(())
}
