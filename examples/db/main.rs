use anyhow::Result;
use usls::{models::DB, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// model file
    #[argh(option)]
    model: Option<String>,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// show bboxes
    #[argh(option, default = "false")]
    show_bboxes: bool,

    /// show mbrs
    #[argh(option, default = "false")]
    show_mbrs: bool,

    /// show bboxes confidence
    #[argh(option, default = "false")]
    show_bboxes_conf: bool,

    /// show mbrs confidence
    #[argh(option, default = "false")]
    show_mbrs_conf: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let options = match &args.model {
        Some(m) => Options::db().with_model_file(m),
        None => Options::ppocr_det_v4_ch().with_model_dtype(args.dtype.as_str().try_into()?),
    };
    let mut model = DB::new(
        options
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
    )?;

    // load image
    let x = DataLoader::try_read_batch(&[
        "images/db.png",
        "images/table.png",
        "images/table-ch.jpg",
        "images/street.jpg",
        "images/slanted-text-number.jpg",
    ])?;

    // run
    let y = model.forward(&x)?;

    // annotate
    let annotator = Annotator::default()
        .without_bboxes(!args.show_bboxes)
        .without_mbrs(!args.show_mbrs)
        .without_bboxes_name(true)
        .without_mbrs_name(true)
        .without_bboxes_conf(!args.show_bboxes_conf)
        .without_mbrs_conf(!args.show_mbrs_conf)
        .with_polygons_alpha(60)
        .with_contours_color([255, 105, 180, 255])
        .with_saveout(model.spec());
    annotator.annotate(&x, &y);

    // summary
    model.summary();

    Ok(())
}
