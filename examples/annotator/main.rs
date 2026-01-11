//! Comprehensive annotation demo showcasing all Drawable types and StyleModes.
//!
//! Output structure:
//! - Hbb/styles.png - All HBB modes + thickness directions in one canvas
//! - Hbb/text_loc.png - All 17 TextLoc positions with boundary cases
//! - Keypoint/styles.png - All keypoint shapes (larger)
//! - Keypoint/skeleton.png - Pose skeleton demo (larger)
//! - Polygon/styles.png - Polygon styles with larger text
//! - Prob/styles.png - Prob positions on single canvas
//! - Mask/styles.png - Mask styles demo
//! - Y/combined.jpg - Combined demo on real image

use clap::{Parser, Subcommand};
use image::{Rgba, RgbaImage};
use usls::{Color, DataLoader};

mod combined;
mod hbb;
mod keypoint;
mod mask;
mod polygon;
mod prob;

#[derive(Parser)]
#[command(author, version, about = "Annotation Examples", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Hbb(hbb::HbbArgs),
    Keypoint(keypoint::KeypointArgs),
    Polygon(polygon::PolygonArgs),
    Prob(prob::ProbArgs),
    Mask(mask::MaskArgs),
    All,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let cli = Cli::parse();
    let bus_image = DataLoader::new("./assets/bus.jpg")?.try_read_one()?;

    match &cli.command {
        Some(Commands::Hbb(_)) => {
            hbb::run(blank_canvas, save_to)?;
        }
        Some(Commands::Keypoint(_)) => {
            keypoint::run(blank_canvas, save_to)?;
        }
        Some(Commands::Polygon(_)) => {
            polygon::run(blank_canvas, save_to)?;
        }
        Some(Commands::Prob(_)) => {
            prob::run(blank_canvas, save_to)?;
        }
        Some(Commands::Mask(_)) => {
            mask::run(blank_canvas, save_to)?;
        }
        Some(Commands::All) | None => {
            hbb::run(blank_canvas, save_to)?;
            keypoint::run(blank_canvas, save_to)?;
            polygon::run(blank_canvas, save_to)?;
            prob::run(blank_canvas, save_to)?;
            mask::run(blank_canvas, save_to)?;
            combined::demo_combined_y(&bus_image)?;
        }
    }

    println!("\n✓ Demos completed! Check runs/Annotate/ for output.");
    Ok(())
}

/// Create a blank canvas with given dimensions and background color
fn blank_canvas(width: u32, height: u32, bg: Color) -> RgbaImage {
    let mut img = RgbaImage::new(width, height);
    let rgba = Rgba(bg.into());
    for pixel in img.pixels_mut() {
        *pixel = rgba;
    }
    img
}

/// Save image to path
fn save_to(img: &RgbaImage, sub_dir: &str, name: &str) -> anyhow::Result<()> {
    let path = usls::Dir::Current
        .base_dir_with_subs(&["runs", "Annotate", sub_dir])?
        .join(format!("{}.png", name));
    img.save(path.display().to_string())?;
    println!("  Saved: {}/{}.png", sub_dir, name);
    Ok(())
}
