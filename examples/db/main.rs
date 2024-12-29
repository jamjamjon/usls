use anyhow::Result;
use usls::{models::DB, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // build model
    let options = Options::ppocr_det_v4_ch().commit()?;
    let mut model = DB::new(options)?;

    // load image
    let x = DataLoader::try_read_batch(&["images/db.png", "images/street.jpg"])?;

    // run
    let y = model.forward(&x)?;

    // annotate
    let annotator = Annotator::default()
        .without_bboxes(true)
        .with_polygons_alpha(60)
        .with_contours_color([255, 105, 180, 255])
        .without_mbrs(true)
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    Ok(())
}
