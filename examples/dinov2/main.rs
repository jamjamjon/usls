use anyhow::Result;
use usls::{models::DINOv2, Config, DataLoader};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // images
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg", "./assets/bus.jpg"])?;

    // model
    let config = Config::dinov2_small()
        .with_batch_size_all(xs.len())
        .commit()?;
    let mut model = DINOv2::new(config)?;

    // encode images
    let y = model.encode_images(&xs)?;
    println!("Feat shape: {:?}", y.shape());

    usls::perf(false);

    Ok(())
}
