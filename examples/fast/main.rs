use anyhow::Result;
use usls::{models::DB, Annotator, Config, DataLoader, Scale};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale
    #[argh(option, default = "String::from(\"t\")")]
    scale: String,

    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build model
    let config = match args.scale.parse()? {
        Scale::T => Config::fast_tiny(),
        Scale::S => Config::fast_small(),
        Scale::B => Config::fast_base(),
        _ => unimplemented!("Unsupported model scale: {:?}. Try b, s, t.", args.scale),
    };
    let mut model = DB::new(
        config
            .with_dtype_all(args.dtype.parse()?)
            .with_device_all(args.device.parse()?)
            .commit()?,
    )?;

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
            usls::PolygonStyle::default()
                .with_visible(true)
                .with_text_visible(false)
                .show_confidence(true)
                .show_id(true)
                .show_name(true)
                .with_outline_color(usls::ColorSource::Custom([255, 105, 180, 255].into())),
        )
        .with_hbb_style(
            usls::HbbStyle::default()
                .with_visible(false)
                .with_text_visible(false)
                .with_thickness(1)
                .show_confidence(false)
                .show_id(false)
                .show_name(false),
        )
        .with_obb_style(
            usls::ObbStyle::default()
                .with_visible(false)
                .with_text_visible(false)
                .show_confidence(false)
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

    usls::perf(false);

    Ok(())
}
