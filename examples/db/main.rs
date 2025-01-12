use anyhow::Result;
use usls::{models::DB, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build model
    let options = Options::ppocr_det_v4_server_ch()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = DB::new(options)?;

    // load image
    let x = DataLoader::try_read_batch(&[
        "images/table.png",
        "images/table1.jpg",
        "images/table2.png",
        "images/table-ch.jpg",
        "images/db.png",
        "images/street.jpg",
    ])?;

    // run
    let y = model.forward(&x)?;

    // annotate
    let annotator = Annotator::default()
        .without_bboxes(true)
        .without_mbrs(true)
        .with_polygons_alpha(60)
        .with_contours_color([255, 105, 180, 255])
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    Ok(())
}
