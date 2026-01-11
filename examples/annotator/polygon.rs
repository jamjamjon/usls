use clap::Parser;
use image::RgbaImage;
use usls::{
    Annotator, Color, ColorSource, Polygon, PolygonStyle, TextLoc, TextStyle, TextStyleMode,
};

#[derive(Parser, Debug)]
pub struct PolygonArgs {}

pub fn run(
    canvas_fn: impl Fn(u32, u32, Color) -> RgbaImage,
    save_fn: impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    demo_polygon_styles(&canvas_fn, &save_fn)?;
    Ok(())
}

fn demo_polygon_styles(
    blank_canvas: &impl Fn(u32, u32, Color) -> RgbaImage,
    save_to: &impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    println!("\n=== Polygon Style Demos ===");

    // Large canvas: 1800x900
    let canvas = blank_canvas(1800, 900, Color::from([30u8, 30, 35, 255])); // Dark background

    let mut polygons = Vec::new();

    // Row 1: Different polygon shapes with vibrant fills
    // 1. Star shape (5-pointed)
    let star_points = create_star_polygon(150.0, 200.0, 80.0, 40.0, 5);
    polygons.push(
        Polygon::try_from(star_points)?
            .with_id(0)
            .with_name("Star")
            .with_confidence(0.95)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([255u8, 215, 0, 180]))) // Gold
                    .with_outline_color(ColorSource::Custom(Color::from([255u8, 140, 0, 255]))) // Dark orange
                    .with_thickness(3)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::OuterTopCenter)
                            .with_font_size(18.0)
                            .with_mode(TextStyleMode::rounded(6.0, 6.0))
                            .with_bg_fill_color(ColorSource::Custom(Color::from([
                                255u8, 215, 0, 220,
                            ]))),
                    ),
            ),
    );

    // 2. Arrow shape
    let arrow = vec![
        [350.0, 180.0],
        [450.0, 120.0],
        [450.0, 160.0],
        [550.0, 160.0],
        [550.0, 240.0],
        [450.0, 240.0],
        [450.0, 280.0],
    ];
    polygons.push(
        Polygon::try_from(arrow)?
            .with_id(1)
            .with_name("Arrow")
            .with_confidence(0.92)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([0u8, 191, 255, 160]))) // Deep sky blue
                    .with_outline_color(ColorSource::Custom(Color::white()))
                    .with_thickness(2)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::Center)
                            .with_font_size(16.0)
                            .with_mode(TextStyleMode::rounded(4.0, 4.0)),
                    ),
            ),
    );

    // 3. Hexagon
    let hexagon = create_regular_polygon(750.0, 200.0, 90.0, 6);
    polygons.push(
        Polygon::try_from(hexagon)?
            .with_id(2)
            .with_name("Hexagon")
            .with_confidence(0.88)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([138u8, 43, 226, 150]))) // Blue violet
                    .with_outline_color(ColorSource::Custom(Color::from([255u8, 0, 255, 255]))) // Magenta
                    .with_thickness(4)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::OuterBottomCenter)
                            .with_font_size(18.0)
                            .with_mode(TextStyleMode::rounded(5.0, 5.0))
                            .with_bg_fill_color(ColorSource::Custom(Color::from([
                                138u8, 43, 226, 200,
                            ]))),
                    ),
            ),
    );

    // 4. Pentagon with thick outline only
    let pentagon = create_regular_polygon(1000.0, 200.0, 85.0, 5);
    polygons.push(
        Polygon::try_from(pentagon)?
            .with_id(3)
            .with_name("Pentagon")
            .with_confidence(0.85)
            .with_style(
                PolygonStyle::default()
                    .with_draw_fill(false)
                    .with_draw_outline(true)
                    .with_outline_color(ColorSource::Custom(Color::from([0u8, 255, 127, 255]))) // Spring green
                    .with_thickness(5)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::Center)
                            .with_font_size(16.0)
                            .with_color(ColorSource::Custom(Color::from([0u8, 255, 127, 255])))
                            .with_bg_fill_color(ColorSource::Custom(Color::from([
                                0u8, 50, 30, 200,
                            ]))),
                    ),
            ),
    );

    // 5. Irregular blob
    let blob = vec![
        [1150.0, 130.0],
        [1250.0, 100.0],
        [1320.0, 150.0],
        [1350.0, 220.0],
        [1300.0, 280.0],
        [1200.0, 300.0],
        [1130.0, 250.0],
        [1100.0, 180.0],
    ];
    polygons.push(
        Polygon::try_from(blob)?
            .with_id(4)
            .with_name("Organic")
            .with_confidence(0.78)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([255u8, 99, 71, 170]))) // Tomato
                    .with_outline_color(ColorSource::Custom(Color::from([255u8, 69, 0, 255]))) // Red-orange
                    .with_thickness(3)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::InnerTopCenter)
                            .with_font_size(15.0),
                    ),
            ),
    );

    // 6. 8-pointed star
    let star8 = create_star_polygon(1550.0, 200.0, 90.0, 45.0, 8);
    polygons.push(
        Polygon::try_from(star8)?
            .with_id(5)
            .with_name("8-Star")
            .with_confidence(0.91)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([255u8, 20, 147, 160]))) // Deep pink
                    .with_outline_color(ColorSource::Custom(Color::white()))
                    .with_thickness(2)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::OuterTopCenter)
                            .with_font_size(16.0)
                            .with_mode(TextStyleMode::rounded(4.0, 4.0)),
                    ),
            ),
    );

    // Row 2: More complex shapes and style combinations
    // 7. Lightning bolt
    let lightning = vec![
        [100.0, 450.0],
        [180.0, 450.0],
        [140.0, 550.0],
        [220.0, 550.0],
        [80.0, 750.0],
        [150.0, 600.0],
        [80.0, 600.0],
    ];
    polygons.push(
        Polygon::try_from(lightning)?
            .with_id(6)
            .with_name("Lightning")
            .with_confidence(0.89)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([255u8, 255, 0, 200]))) // Yellow
                    .with_outline_color(ColorSource::Custom(Color::from([255u8, 200, 0, 255])))
                    .with_thickness(3)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::OuterTopLeft)
                            .with_font_size(14.0)
                            .with_mode(TextStyleMode::rounded(3.0, 3.0)),
                    ),
            ),
    );

    // 8. House shape
    let house = vec![
        [350.0, 550.0],
        [500.0, 450.0],
        [650.0, 550.0],
        [620.0, 550.0],
        [620.0, 750.0],
        [380.0, 750.0],
        [380.0, 550.0],
    ];
    polygons.push(
        Polygon::try_from(house)?
            .with_id(7)
            .with_name("House")
            .with_confidence(0.94)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([139u8, 69, 19, 180]))) // Saddle brown
                    .with_outline_color(ColorSource::Custom(Color::from([210u8, 180, 140, 255]))) // Tan
                    .with_thickness(4)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::InnerBottomCenter)
                            .with_font_size(18.0)
                            .with_bg_fill_color(ColorSource::Custom(Color::from([
                                139u8, 69, 19, 220,
                            ]))),
                    ),
            ),
    );

    // 9. Crescent/moon shape
    let crescent = vec![
        [800.0, 500.0],
        [850.0, 450.0],
        [920.0, 450.0],
        [970.0, 500.0],
        [970.0, 600.0],
        [920.0, 650.0],
        [850.0, 650.0],
        [800.0, 600.0],
        [830.0, 550.0],
        [850.0, 520.0],
        [880.0, 520.0],
        [900.0, 550.0],
        [900.0, 570.0],
        [880.0, 590.0],
        [850.0, 590.0],
        [830.0, 570.0],
    ];
    polygons.push(
        Polygon::try_from(crescent)?
            .with_id(8)
            .with_name("Ring")
            .with_confidence(0.82)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([100u8, 149, 237, 180]))) // Cornflower blue
                    .with_outline_color(ColorSource::Custom(Color::from([65u8, 105, 225, 255]))) // Royal blue
                    .with_thickness(3)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::Center)
                            .with_font_size(16.0),
                    ),
            ),
    );

    // 10. Triangle with gradient-like effect
    let triangle = create_regular_polygon(1150.0, 600.0, 100.0, 3);
    polygons.push(
        Polygon::try_from(triangle)?
            .with_id(9)
            .with_name("Triangle")
            .with_confidence(0.96)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([50u8, 205, 50, 160]))) // Lime green
                    .with_outline_color(ColorSource::Custom(Color::from([34u8, 139, 34, 255]))) // Forest green
                    .with_thickness(5)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::OuterBottomCenter)
                            .with_font_size(17.0)
                            .with_mode(TextStyleMode::rounded(5.0, 5.0))
                            .with_bg_fill_color(ColorSource::Custom(Color::from([
                                50u8, 205, 50, 200,
                            ]))),
                    ),
            ),
    );

    // 11. Cross shape
    let cross = vec![
        [1350.0, 450.0],
        [1400.0, 450.0],
        [1400.0, 520.0],
        [1470.0, 520.0],
        [1470.0, 570.0],
        [1400.0, 570.0],
        [1400.0, 640.0],
        [1350.0, 640.0],
        [1350.0, 570.0],
        [1280.0, 570.0],
        [1280.0, 520.0],
        [1350.0, 520.0],
    ];
    polygons.push(
        Polygon::try_from(cross)?
            .with_id(10)
            .with_name("Cross")
            .with_confidence(0.87)
            .with_style(
                PolygonStyle::default()
                    .with_fill_color(ColorSource::Custom(Color::from([220u8, 20, 60, 180]))) // Crimson
                    .with_outline_color(ColorSource::Custom(Color::white()))
                    .with_thickness(2)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::OuterTopCenter)
                            .with_font_size(15.0)
                            .with_mode(TextStyleMode::rounded(4.0, 4.0)),
                    ),
            ),
    );

    // 12. 12-pointed star with outline only
    let star12 = create_star_polygon(1650.0, 600.0, 100.0, 60.0, 12);
    polygons.push(
        Polygon::try_from(star12)?
            .with_id(11)
            .with_name("12-Star")
            .with_confidence(0.93)
            .with_style(
                PolygonStyle::default()
                    .with_draw_fill(false)
                    .with_draw_outline(true)
                    .with_outline_color(ColorSource::Custom(Color::from([255u8, 165, 0, 255]))) // Orange
                    .with_thickness(4)
                    .with_text_visible(true)
                    .with_text_style(
                        TextStyle::default()
                            .with_loc(TextLoc::Center)
                            .with_font_size(14.0)
                            .with_color(ColorSource::Custom(Color::from([255u8, 165, 0, 255])))
                            .with_draw_fill(false),
                    ),
            ),
    );

    let annotator = Annotator::default();
    let result = annotator.annotate(&canvas.into(), &polygons)?;
    save_to(&result.into(), "Polygon", "styles")?;

    Ok(())
}

/// Create a star polygon
fn create_star_polygon(
    cx: f32,
    cy: f32,
    outer_r: f32,
    inner_r: f32,
    points: usize,
) -> Vec<[f32; 2]> {
    let mut vertices = Vec::with_capacity(points * 2);
    for i in 0..(points * 2) {
        let angle =
            (i as f32) * std::f32::consts::PI / (points as f32) - std::f32::consts::PI / 2.0;
        let r = if i % 2 == 0 { outer_r } else { inner_r };
        vertices.push([cx + r * angle.cos(), cy + r * angle.sin()]);
    }
    vertices
}

/// Create a regular polygon
fn create_regular_polygon(cx: f32, cy: f32, radius: f32, sides: usize) -> Vec<[f32; 2]> {
    let mut vertices = Vec::with_capacity(sides);
    for i in 0..sides {
        let angle =
            (i as f32) * 2.0 * std::f32::consts::PI / (sides as f32) - std::f32::consts::PI / 2.0;
        vertices.push([cx + radius * angle.cos(), cy + radius * angle.sin()]);
    }
    vertices
}
