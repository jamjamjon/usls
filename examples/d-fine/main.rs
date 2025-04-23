use anyhow::Result;
use usls::{models::RTDETR, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // options
    let options = Options::d_fine_n_coco().commit()?;
    let mut model = RTDETR::new(options)?;

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

    Ok(())
}
