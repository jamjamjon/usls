use usls::{models::MODNet, Annotator, DataLoader, Options};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // build model
    let options = Options::modnet_photographic().commit()?;
    let mut model = MODNet::new(options)?;

    // load image
    let xs = DataLoader::try_read_n(&["images/liuyifei.png"])?;

    // run
    let ys = model.forward(&xs)?;

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
