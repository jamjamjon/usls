use anyhow::Result;
use usls::{models::RTDETR, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // options
    let options = Options::dfine_n_coco().commit()?;
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
