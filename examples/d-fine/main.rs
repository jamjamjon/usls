use anyhow::Result;
use usls::{models::RTDETR, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // options
    let options = Options::d_fine_n_coco().commit()?;
    let mut model = RTDETR::new(options)?;

    // load
    let x = [DataLoader::try_read("./assets/bus.jpg")?];

    // run
    let y = model.forward(&x)?;
    println!("{:?}", y);

    // annotate
    let annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    Ok(())
}
