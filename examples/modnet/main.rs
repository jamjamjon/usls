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
    let xs = [DataLoader::try_read("images/liuyifei.png")?];

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default().with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    Ok(())
}
