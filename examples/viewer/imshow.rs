use clap::Parser;
use usls::{DataLoader, Source, Viewer};

#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Data source.
    #[arg(long, required = true)]
    source: Source,

    /// Save frames to video.
    #[arg(long, default_value = "false")]
    save: bool,

    /// Window scale.
    #[arg(long, default_value = "0.8")]
    window_scale: f32,

    /// Delay in milliseconds between frames.
    #[arg(long, default_value = "1")]
    delay: u64,

    /// Num of frames to skip.
    #[arg(long, default_value = "0")]
    nfv_skip: u64,
}

fn main() -> anyhow::Result<()> {
    utils::init_logging();
    let args = Args::parse();
    let dl = DataLoader::new(args.source)?
        .with_nfv_skip(args.nfv_skip)
        .stream()?;
    let mut viewer = Viewer::default().with_window_scale(args.window_scale);

    for images in &dl {
        // Check if the window is closed and exit if so.
        if viewer.is_window_exist_and_closed() {
            break;
        }

        // Display the current image.
        viewer.imshow(&images[0])?;

        // Wait for a key press or timeout, and exit on Escape.
        if let Some(key) = viewer.wait_key(args.delay) {
            if key == usls::Key::Escape {
                break;
            }
        }

        // Save the current frame to video if requested.
        // Note: For multiple videos, frames will be saved to separate files.
        if args.save {
            viewer.write_video_frame(&images[0])?;
        }
    }

    Ok(())
}
