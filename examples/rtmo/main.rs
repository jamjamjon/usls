use anyhow::Result;
use usls::{models::RTMO, Annotator, DataLoader, Options, Style, SKELETON_COCO_19};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // build model
    let mut model = RTMO::new(Options::rtmo_s().commit()?)?;

    // load image
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;
    println!("ys: {:?}", ys);

    // annotate
    let annotator = Annotator::default()
        .with_hbb_style(Style::hbb().with_draw_fill(true))
        .with_keypoint_style(
            Style::keypoint()
                .with_skeleton(SKELETON_COCO_19.into())
                .show_confidence(false)
                .show_id(true)
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

    Ok(())
}
