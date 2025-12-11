use anyhow::Result;
use usls::{models::YOLOPv2, Annotator, Config, DataLoader, PolygonStyle};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // build model
    let mut model = YOLOPv2::new(Config::yolop_v2_480x800().commit()?)?;

    // load image
    let xs = DataLoader::try_read_n(&["images/car-view.jpg"])?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default()
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(true)
                .with_cutout(true)
                .with_draw_polygon_largest(true),
        )
        .with_polygon_style(
            PolygonStyle::default()
                .with_text_visible(true)
                .show_name(true)
                .show_id(true),
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
