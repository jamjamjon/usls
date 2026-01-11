use clap::Parser;
use image::RgbaImage;
use usls::{Annotator, Color, Prob, ProbStyle, TextLoc, TextStyle, TextStyleMode};

#[derive(Parser, Debug)]
pub struct ProbArgs {}

pub fn run(
    canvas_fn: impl Fn(u32, u32, Color) -> RgbaImage,
    save_fn: impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    demo_prob_styles(&canvas_fn, &save_fn)?;
    Ok(())
}

fn demo_prob_styles(
    blank_canvas: &impl Fn(u32, u32, Color) -> RgbaImage,
    save_to: &impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    println!("\n=== Prob Style Demos ===");

    // Larger canvas: 1200x900 to show more positions and font size variations
    let canvas = blank_canvas(1200, 900, Color::from([250u8, 250, 250, 255]));

    // Create probs at different locations with VARYING font sizes
    let probs = vec![
        // Top row - varying sizes
        Prob::default()
            .with_id(0)
            .with_name("TopLeft(16px)")
            .with_confidence(0.95)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerTopLeft)
                        .with_font_size(16.0),
                ),
            ),
        Prob::default()
            .with_id(1)
            .with_name("TopCenter(24px)")
            .with_confidence(0.88)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerTopCenter)
                        .with_font_size(24.0),
                ),
            ),
        Prob::default()
            .with_id(2)
            .with_name("TopRight(32px)")
            .with_confidence(0.82)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerTopRight)
                        .with_font_size(32.0),
                ),
            ),
        // Center row - different styles
        Prob::default()
            .with_id(3)
            .with_name("CenterLeft(20px)")
            .with_confidence(0.75)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerCenterLeft)
                        .with_font_size(20.0),
                ),
            ),
        Prob::default()
            .with_id(4)
            .with_name("Center(28px)")
            .with_confidence(0.99)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::Center)
                        .with_font_size(28.0)
                        .with_mode(TextStyleMode::rounded(6.0, 4.0)),
                ),
            ),
        Prob::default()
            .with_id(5)
            .with_name("CenterRight(22px)")
            .with_confidence(0.68)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerCenterRight)
                        .with_font_size(22.0),
                ),
            ),
        // Bottom row - with outlines
        Prob::default()
            .with_id(6)
            .with_name("BotLeft(18px)")
            .with_confidence(0.55)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerBottomLeft)
                        .with_font_size(18.0)
                        .with_draw_outline(true)
                        .with_thickness(2),
                ),
            ),
        Prob::default()
            .with_id(7)
            .with_name("BotCenter(26px)")
            .with_confidence(0.45)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerBottomCenter)
                        .with_font_size(26.0)
                        .with_mode(TextStyleMode::rounded(8.0, 5.0))
                        .with_draw_outline(true)
                        .with_thickness(2),
                ),
            ),
        Prob::default()
            .with_id(8)
            .with_name("BotRight(36px)")
            .with_confidence(0.35)
            .with_style(
                ProbStyle::default().with_text_style(
                    TextStyle::default()
                        .with_loc(TextLoc::InnerBottomRight)
                        .with_font_size(36.0),
                ),
            ),
    ];

    let annotator = Annotator::default();
    let mut result: RgbaImage = canvas;

    // Draw each prob individually since they have different styles
    for prob in &probs {
        let temp = annotator.annotate(&result.clone().into(), prob)?;
        result = temp.into();
    }

    save_to(&result, "Prob", "styles")?;

    Ok(())
}
