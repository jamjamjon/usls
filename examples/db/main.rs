use anyhow::Result;
use usls::{models::DB, Annotator, Config, DataLoader, Style};

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

    /// show hbbs
    #[argh(option, default = "false")]
    show_hbbs: bool,

    /// show obbs
    #[argh(option, default = "false")]
    show_obbs: bool,

    /// show bboxes confidence
    #[argh(option, default = "false")]
    show_hbbs_conf: bool,

    /// show obbs confidence
    #[argh(option, default = "false")]
    show_obbs_conf: bool,

    /// show polygons confidence
    #[argh(option, default = "false")]
    show_polygons_conf: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = match &args.model {
        Some(m) => Config::db().with_model_file(m),
        None => Config::ppocr_det_v5_mobile().with_model_dtype(args.dtype.parse()?),
    }
    .with_device_all(args.device.parse()?)
    .commit()?;
    let mut model = DB::new(config)?;

    // load image
    let xs = DataLoader::try_read_n(&[
        "images/db.png",
        "images/table.png",
        "images/table-ch.jpg",
        "images/street.jpg",
        "images/slanted-text-number.jpg",
    ])?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default()
        .with_polygon_style(
            Style::polygon()
                .with_visible(true)
                .with_text_visible(true)
                .show_confidence(args.show_polygons_conf)
                .show_id(false)
                .show_name(false)
                .with_color(usls::StyleColors::default().with_outline([255, 105, 180, 255].into())),
        )
        .with_hbb_style(
            Style::hbb()
                .with_visible(args.show_hbbs)
                .with_text_visible(true)
                .with_thickness(1)
                .show_confidence(args.show_hbbs_conf)
                .show_id(false)
                .show_name(false),
        )
        .with_obb_style(
            Style::obb()
                .with_visible(args.show_obbs)
                .with_text_visible(true)
                .show_confidence(args.show_obbs_conf)
                .show_id(false)
                .show_name(false),
        );

    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", model.spec()])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    // summary
    usls::perf(false);

    Ok(())
}
