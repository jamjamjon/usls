fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // build
    let mut hub = usls::Hub::new()?;
    println!("{:#?}", hub);

    // download
    let image_downloaded = hub.try_fetch("images/bus.jpg")?;
    println!("Fetch one image. path: {:?}", image_downloaded);

    // download again
    let image_downloaded = hub.try_fetch("images/bus.jpg")?;
    println!("Fetch one image. path: {:?}", image_downloaded);

    // tags and files
    for tag in hub.tags().iter() {
        let files = hub.files(tag);
        println!("{} => {:?}", tag, files);
    }

    Ok(())
}
