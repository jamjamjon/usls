use anyhow::Result;
use usls::{models::RTDETR, Annotator, Config, DataLoader};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // config
    let config = Config::rtdetr_v2_s_coco().commit()?;
    // rtdetr_v1_r18vd_coco()
    // rtdetr_v2_ms_coco()
    // rtdetr_v2_m_coco()
    // rtdetr_v2_l_coco()
    // rtdetr_v2_x_coco()
    let mut model = RTDETR::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default();
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
