use usls::{DataLoader, Image, ImageVecExt};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // 1. Read one image
    let image = Image::try_read("./assets/bus.jpg")?;
    println!("Image::try_read(): {:?}", image);
    // image.save("kkk.png")?;

    // => To Rgba8
    let _image_rgba = image.to_rgba8();

    // 2. Read one image with DataLoader
    let image = DataLoader::try_read_one("./assets/bus.jpg")?;
    println!("DataLoader::try_read_one(): {:?}", image);

    // 3. Read N images with DataLoader
    let images = DataLoader::try_read_n(&["./assets/bus.jpg", "./assets/cat.png"])?;
    println!("DataLoader::try_read_n():");
    for image in images {
        println!(" - {:?}", image);
    }

    // 4. Read image folder with DataLoader
    let images = DataLoader::try_read_folder("./assets")?;
    println!("DataLoader::try_read_folder():");
    for image in images {
        println!(" - {:?}", image);
    }

    // 5. Glob and read image folder with DataLoader
    // let images = DataLoader::try_read_pattern("./assets/*.Jpg")?;
    let images = DataLoader::try_read_pattern_case_insensitive("./assets/*.Jpg")?;
    println!("DataLoader::try_read_pattern_case_insensitive():");

    for image in images {
        println!(" - {:?}", image);
    }

    // 6. Load images with DataLoader
    let dl = DataLoader::new("./assets")?.with_batch(2).build()?;

    // iterate over the dataloader
    for (i, images) in dl.iter().enumerate() {
        println!("## Batch-{}: {:?}", i + 1, images);
    }

    // 7. Vec<Image> <-> Vec<DynamicImage>
    let images = DataLoader::try_read_n(&["./assets/bus.jpg", "./assets/cat.png"])?;
    let dyn_images = images.into_dyns();
    let _images = dyn_images.into_images();

    Ok(())
}
