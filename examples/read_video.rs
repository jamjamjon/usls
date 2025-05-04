use usls::DataLoader;

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// source
    #[argh(
        option,
        default = "String::from(\"http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4\")"
    )]
    source: String,
}

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // load images or video stream
    let dl = DataLoader::new(args.source.as_str())?
        .with_batch(1)
        // .with_nf_skip(1)
        // .with_progress_bar(true)
        .build()?;

    // iterate over the dataloader
    for images in &dl {
        for image in &images {
            println!("## {:?}", image);
        }
    }

    Ok(())
}
