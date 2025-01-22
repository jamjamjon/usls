use usls::Hub;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // 1. Download from default github release
    let path = Hub::default().try_fetch("images/bus.jpg")?;
    println!("Fetch one image: {:?}", path);

    // 2. Download from specific github release url
    let path = Hub::default()
        .try_fetch("https://github.com/jamjamjon/assets/releases/download/images/bus.jpg")?;
    println!("Fetch one file: {:?}", path);

    // 3. Fetch tags and files
    let hub = Hub::default().with_owner("jamjamjon").with_repo("usls");
    for (i, tag) in hub.tags().iter().enumerate() {
        let files = hub.files(tag);
        println!("{} :: {} => {:?}", i, tag, files); // Should be empty
    }

    Ok(())
}
