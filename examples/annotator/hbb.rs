use clap::Parser;
use image::RgbaImage;
use usls::{
    Annotator, Color, ColorSource, Hbb, HbbStyle, TextLoc, TextStyle, TextStyleMode,
    ThicknessDirection,
};

#[derive(Parser, Debug)]
pub struct HbbArgs {}

pub fn run(
    canvas_fn: impl Fn(u32, u32, Color) -> RgbaImage,
    save_fn: impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    demo_hbb_styles(&canvas_fn, &save_fn)?;
    demo_hbb_text_locations(&canvas_fn, &save_fn)?;
    Ok(())
}

fn demo_hbb_styles(
    blank_canvas: &impl Fn(u32, u32, Color) -> RgbaImage,
    save_to: &impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    println!("\n=== HBB Styles Demo ===");

    // Large canvas: 1200x800
    let canvas = blank_canvas(1200, 800, Color::from([245u8, 245, 245, 255]));

    // Row 1: Style modes (Solid, Dashed, Corners, Rounded)
    let modes: Vec<(&str, HbbStyle)> = vec![
        (
            "Solid",
            HbbStyle::default()
                .with_thickness(4)
                .with_draw_fill(true)
                .with_fill_color(ColorSource::AutoAlpha(60)),
        ),
        ("Dashed", HbbStyle::dashed().with_thickness(4)),
        ("Corners", HbbStyle::corners().with_thickness(6)),
        (
            "Rounded",
            HbbStyle::rounded()
                .with_thickness(4)
                .with_draw_fill(true)
                .with_fill_color(ColorSource::AutoAlpha(60))
                .with_text_style(
                    TextStyle::default()
                        .with_font_size(28.0)
                        .with_mode(TextStyleMode::rounded(8.0, 8.0)) // Rounded corners for text box
                        .with_bg_fill_color(ColorSource::Custom(Color::yellow()))
                        .with_bg_outline_color(ColorSource::Custom(Color::black()))
                        .with_draw_outline(true)
                        .with_thickness(2),
                ),
        ),
    ];

    // Row 2: Thickness directions
    let directions: Vec<(&str, ThicknessDirection, Color)> = vec![
        ("Outward", ThicknessDirection::Outward, Color::red()),
        ("Inward", ThicknessDirection::Inward, Color::green()),
        ("Centered", ThicknessDirection::Centered, Color::blue()),
    ];

    let mut hbbs = Vec::new();

    // Style modes in row 1
    for (i, (name, style)) in modes.iter().enumerate() {
        let x = 50.0 + i as f32 * 280.0;
        hbbs.push(
            Hbb::default()
                .with_xyxy(x, 80.0, x + 230.0, 320.0)
                .with_id(i)
                .with_name(name)
                .with_confidence(0.95)
                .with_style(style.clone()),
        );
    }

    // Thickness directions in row 2 with labels
    for (i, (name, dir, color)) in directions.iter().enumerate() {
        let x = 100.0 + i as f32 * 350.0;
        let thickness = 3 + 8 * i;

        // Add the main box with thickness direction
        hbbs.push(
            Hbb::default()
                .with_xyxy(x, 450.0, x + 280.0, 700.0)
                .with_id(i + 4)
                .with_name(&format!("thickness={}", thickness))
                .with_confidence(0.88)
                .with_style(
                    HbbStyle::default()
                        .with_thickness(thickness)
                        .with_thickness_direction(*dir)
                        .with_outline_color(ColorSource::Custom(*color)),
                ),
        );

        // Add thickness label
        let label = format!("Direction: {}", name);
        hbbs.push(
            Hbb::default()
                .with_xyxy(x + 10.0, 720.0, x + 270.0, 750.0)
                .with_id(i + 10)
                .with_name(&label)
                .with_style(
                    HbbStyle::default()
                        .with_draw_outline(false)
                        .with_text_style(
                            TextStyle::default()
                                .with_font_size(20.0)
                                .with_loc(TextLoc::Center)
                                .with_bg_fill_color(ColorSource::Custom(Color::white()))
                                .with_bg_outline_color(ColorSource::Custom(Color::black()))
                                .with_draw_outline(true)
                                .with_thickness(1),
                        ),
                ),
        );
    }

    let annotator = Annotator::default().with_hbb_style(
        HbbStyle::default().with_text_style(TextStyle::default().with_font_size(28.0)),
    );
    let result = annotator.annotate(&canvas.into(), &hbbs)?;
    save_to(&result.into(), "Hbb", "styles")?;

    Ok(())
}

fn demo_hbb_text_locations(
    blank_canvas: &impl Fn(u32, u32, Color) -> RgbaImage,
    save_to: &impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    println!("\n=== HBB TextLoc Demo ===");

    // Large canvas for boundary cases: 1800x1400
    let canvas = blank_canvas(1800, 1400, Color::from([250u8, 250, 250, 255]));

    // Center box (large enough for long text labels)
    let center_box = (300.0, 280.0, 1500.0, 1120.0);

    // Define all 17 positions with their expected visual locations
    let text_locs = [
        // Outer top row
        (TextLoc::OuterTopLeft, "OuterTopLeft"),
        (TextLoc::OuterTopCenter, "OuterTopCenter"),
        (TextLoc::OuterTopRight, "OuterTopRight"),
        // Inner top row
        (TextLoc::InnerTopLeft, "InnerTopLeft"),
        (TextLoc::InnerTopCenter, "InnerTopCenter"),
        (TextLoc::InnerTopRight, "InnerTopRight"),
        // Left/Right sides
        (TextLoc::OuterCenterLeft, "OuterCenterLeft"),
        (TextLoc::InnerCenterLeft, "InnerCenterLeft"),
        (TextLoc::Center, "Center"),
        (TextLoc::InnerCenterRight, "InnerCenterRight"),
        (TextLoc::OuterCenterRight, "OuterCenterRight"),
        // Inner bottom row
        (TextLoc::InnerBottomLeft, "InnerBottomLeft"),
        (TextLoc::InnerBottomCenter, "InnerBottomCenter"),
        (TextLoc::InnerBottomRight, "InnerBottomRight"),
        // Outer bottom row
        (TextLoc::OuterBottomLeft, "OuterBottomLeft"),
        (TextLoc::OuterBottomCenter, "OuterBottomCenter"),
        (TextLoc::OuterBottomRight, "OuterBottomRight"),
    ];

    let mut hbbs = Vec::new();

    // Main center box with all text positions
    for (i, (loc, name)) in text_locs.iter().enumerate() {
        hbbs.push(
            Hbb::default()
                .with_xyxy(center_box.0, center_box.1, center_box.2, center_box.3)
                .with_id(i)
                .with_name(name)
                .with_confidence(0.99)
                .with_style(
                    HbbStyle::default()
                        .with_thickness(3)
                        .with_draw_outline(i == 0) // Only draw outline for first one
                        .with_draw_fill(false)
                        .with_text_style(TextStyle::default().with_loc(*loc).with_font_size(20.0)),
                ),
        );
    }

    // Boundary case boxes - at canvas edges
    // Top-left corner box
    hbbs.push(
        Hbb::default()
            .with_xyxy(0.0, 0.0, 150.0, 100.0)
            .with_id(100)
            .with_name("EdgeTopLeft")
            .with_confidence(0.88)
            .with_style(
                HbbStyle::default()
                    .with_thickness(2)
                    .with_outline_color(ColorSource::Custom(Color::red()))
                    .with_text_style(TextStyle::default().with_font_size(18.0)),
            ),
    );

    // Top-right corner box
    hbbs.push(
        Hbb::default()
            .with_xyxy(1650.0, 0.0, 1800.0, 100.0)
            .with_id(101)
            .with_name("EdgeTopRight")
            .with_confidence(0.77)
            .with_style(
                HbbStyle::default()
                    .with_thickness(2)
                    .with_outline_color(ColorSource::Custom(Color::red()))
                    .with_text_style(TextStyle::default().with_font_size(18.0)),
            ),
    );

    // Bottom-left corner box
    hbbs.push(
        Hbb::default()
            .with_xyxy(0.0, 1300.0, 150.0, 1400.0)
            .with_id(2)
            .with_name("EdgeBotLeft")
            .with_confidence(0.66)
            .with_style(
                HbbStyle::default()
                    .with_thickness(2)
                    .with_outline_color(ColorSource::Custom(Color::red()))
                    .with_text_style(TextStyle::default().with_font_size(18.0)),
            ),
    );

    // Bottom-right corner box
    hbbs.push(
        Hbb::default()
            .with_xyxy(1650.0, 1300.0, 1800.0, 1400.0)
            .with_id(103)
            .with_name("EdgeBotRight")
            .with_confidence(0.55)
            .with_style(
                HbbStyle::default()
                    .with_thickness(2)
                    .with_outline_color(ColorSource::Custom(Color::red()))
                    .with_text_style(TextStyle::default().with_font_size(18.0)),
            ),
    );

    // Tiny box (text larger than box)
    hbbs.push(
        Hbb::default()
            .with_xyxy(50.0, 150.0, 80.0, 180.0)
            .with_id(104)
            .with_name("TinyBox")
            .with_confidence(0.99)
            .with_style(
                HbbStyle::default()
                    .with_thickness(2)
                    .with_outline_color(ColorSource::Custom(Color::magenta()))
                    .with_text_style(TextStyle::default().with_font_size(18.0)),
            ),
    );

    let annotator = Annotator::default();
    let result = annotator.annotate(&canvas.into(), &hbbs)?;
    save_to(&result.into(), "Hbb", "text_loc")?;

    Ok(())
}
