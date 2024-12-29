use usls::DataLoader;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // 1. iterator
    let dl = DataLoader::try_from(
        // "images/bus.jpg", // remote image
        // "../images", // image folder
        // "../demo.mp4",   // local video
        // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4", // remote video
        // "rtsp://admin:xyz@192.168.2.217:554/h265/ch1/",  // rtsp h264 stream
        "./assets/bus.jpg", // local image
    )?
    .with_batch(1)
    .with_progress_bar(true)
    .build()?;

    for (_xs, _paths) in dl {
        println!("Paths: {:?}", _paths);
    }

    // 2. read one image
    let image = DataLoader::try_read("./assets/bus.jpg")?;
    println!(
        "Read one image. Height: {}, Width: {}",
        image.height(),
        image.width()
    );

    // 3. read several images
    let images = DataLoader::try_read_batch(&[
        "./assets/bus.jpg",
        "./assets/bus.jpg",
        "./assets/bus.jpg",
        "./assets/bus.jpg",
        "./assets/bus.jpg",
    ])?;
    println!("Read {} images.", images.len());

    Ok(())
}
