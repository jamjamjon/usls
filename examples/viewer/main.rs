use usls::{DataLoader, Key, Viewer};

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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();
    let dl = DataLoader::new(&args.source)?.with_batch(1).build()?;

    let mut viewer = Viewer::new().with_delay(5).with_scale(1.).resizable(true);

    // run & annotate
    for (xs, _paths) in dl {
        // show image
        viewer.imshow(&xs)?;

        // check out window and key event
        if !viewer.is_open() || viewer.is_key_pressed(Key::Escape) {
            break;
        }

        // write video
        viewer.write_batch(&xs)?
    }

    // finish video write
    viewer.finish_write()?;

    Ok(())
}
