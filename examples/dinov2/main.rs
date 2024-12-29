use anyhow::Result;
use usls::{models::DINOv2, DataLoader, Options};

fn main() -> Result<()> {
    // images
    let xs = [
        DataLoader::try_read("./assets/bus.jpg")?,
        DataLoader::try_read("./assets/bus.jpg")?,
    ];

    // model
    let options = Options::dinov2_small().with_batch_size(xs.len()).commit()?;
    let mut model = DINOv2::new(options)?;

    // encode images
    let y = model.encode_images(&xs)?;
    println!("Feat shape: {:?}", y.shape());

    Ok(())
}
