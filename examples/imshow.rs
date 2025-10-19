use usls::{DataLoader, Viewer};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// source
    #[argh(option, default = "String::from(\"./assets\")")]
    source: String,
    // /// record video and save
    // #[argh(option, default = "false")]
    // save_video: bool,
}

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let dl = DataLoader::new(args.source.as_str())?.build()?;
    let mut viewer = Viewer::default().with_window_scale(1.);

    for images in &dl {
        // check out window
        if viewer.is_window_exist() && !viewer.is_window_open() {
            break;
        }

        viewer.imshow(&images[0])?;

        // check out key event
        if let Some(key) = viewer.wait_key(1000) {
            if key == usls::Key::Escape {
                break;
            }
        }

        // image info
        for image in &images {
            println!("## {:?}", image);
        }

        // // write video, need  video feature
        // if args.save_video {
        //     viewer.write_video_frame(&images[0])?;
        // }
    }

    Ok(())
}
