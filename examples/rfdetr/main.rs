use anyhow::Result;
use usls::{models::RFDETR, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // options
    let options = Options::rfdetr_base().commit()?;
    let mut model = RFDETR::new(options)?;

    // load
    let xs = [DataLoader::try_read("./assets/bus.jpg")?];

    // run
    let ys = model.forward(&xs)?;

    // extract bboxes
    for y in ys.iter() {
        if let Some(bboxes) = y.bboxes() {
            println!("[Bboxes]: Found {} objects", bboxes.len());
            for (i, bbox) in bboxes.iter().enumerate() {
                println!("{}: {:?}", i, bbox)
            }
        }
    }

    // annotate
    let annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    Ok(())
}
