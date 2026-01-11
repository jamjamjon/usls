use clap::Parser;
use image::{GrayImage, Luma, RgbaImage};
use usls::{Annotator, Color, Mask, MaskStyle, MaskStyleMode};

#[derive(Parser, Debug)]
pub struct MaskArgs {}

pub fn run(
    canvas_fn: impl Fn(u32, u32, Color) -> RgbaImage,
    save_fn: impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    demo_mask_styles(&canvas_fn, &save_fn)?;
    Ok(())
}

fn demo_mask_styles(
    blank_canvas: &impl Fn(u32, u32, Color) -> RgbaImage,
    save_to: &impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    println!("\n=== Mask Style Demos ===");

    // Canvas size: 1200x500
    let canvas = blank_canvas(1200, 500, Color::from([240u8, 240, 240, 255]));

    // Create sample masks (circles at different positions)
    let positions = [
        (200u32, 250u32, 120u32, "Overlay"),
        (600, 250, 100, "Halo Purple"),
        (1000, 250, 90, "Halo Custom"),
    ];

    let styles: Vec<MaskStyle> = vec![
        MaskStyle::default()
            .with_cutout(false)
            .with_palette(&[Color::red().with_alpha(128)]),
        MaskStyle::halo()
            .with_cutout(false)
            .with_palette(&[Color::cyan()]),
        MaskStyle::default()
            .with_mode(MaskStyleMode::halo_with(
                0.08,
                Color::magenta().with_alpha(200),
            ))
            .with_cutout(false)
            .with_palette(&[Color::green()]),
    ];

    let mut result: RgbaImage = canvas;

    for (i, ((cx, cy, radius, name), style)) in positions.iter().zip(styles.iter()).enumerate() {
        // Create circular mask
        let mut gray = GrayImage::new(1200, 500);
        for y in 0..500 {
            for x in 0..1200 {
                let dx = x as i32 - *cx as i32;
                let dy = y as i32 - *cy as i32;
                if (dx * dx + dy * dy) <= (*radius as i32 * *radius as i32) {
                    gray.put_pixel(x, y, Luma([255u8]));
                }
            }
        }

        let mask = Mask::new(&gray.into_raw(), 1200, 500)?
            .with_id(i)
            .with_name(name)
            .with_confidence(0.95)
            .with_style(style.clone());

        let annotator = Annotator::default();
        let temp = annotator.annotate(&result.clone().into(), &mask)?;
        result = temp.into();
    }

    save_to(&result, "Mask", "styles")?;

    Ok(())
}
