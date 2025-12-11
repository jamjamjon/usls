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

use image::{GrayImage, Luma, Rgba, RgbaImage};
use usls::{
    Annotator, Color, ColorSource, DataLoader, Hbb, HbbStyle, Keypoint, KeypointStyle,
    KeypointStyleMode, Mask, MaskStyle, MaskStyleMode, Polygon, PolygonStyle, Prob, ProbStyle,
    Skeleton, TextLoc, TextStyle, TextStyleMode, ThicknessDirection, SKELETON_COCO_19,
    SKELETON_COLOR_COCO_19, Y,
};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let bus_image = DataLoader::try_read_one("./assets/bus.jpg")?;
    println!("Loaded bus.jpg: {:?}", bus_image.dimensions());

    demo_hbb_styles()?;
    demo_hbb_text_locations()?;
    demo_keypoint_styles()?;
    demo_keypoint_skeleton()?;
    demo_polygon_styles()?;
    demo_prob_styles()?;
    demo_mask_styles()?;
    demo_combined_y(&bus_image)?;

    println!("\n✓ All demos completed! Check runs/Annotate/ for output.");
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

// =============================================================================
// HBB Styles Demo - All modes + thickness directions on one canvas
// =============================================================================
fn demo_hbb_styles() -> anyhow::Result<()> {
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
        let thickness = 20;

        // Add the main box with thickness direction
        hbbs.push(
            Hbb::default()
                .with_xyxy(x, 450.0, x + 280.0, 700.0)
                .with_id(i + 4)
                .with_name(name)
                .with_confidence(0.88)
                .with_style(
                    HbbStyle::default()
                        .with_thickness(thickness)
                        .with_thickness_direction(*dir)
                        .with_outline_color(ColorSource::Custom(*color)),
                ),
        );

        // Add thickness label
        let label = format!("Thickness: {}", thickness);
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

// =============================================================================
// HBB TextLoc Demo - All 17 positions with boundary cases
// =============================================================================
fn demo_hbb_text_locations() -> anyhow::Result<()> {
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
            .with_id(102)
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

// =============================================================================
// Keypoint Styles Demo - All shapes on large canvas
// =============================================================================
fn demo_keypoint_styles() -> anyhow::Result<()> {
    println!("\n=== Keypoint Styles Demo ===");

    // Large canvas: 1400x600
    let canvas = blank_canvas(1400, 600, Color::from([250u8, 250, 250, 255]));

    // Use explicit colors for each mode to ensure visibility
    let modes: Vec<(&str, KeypointStyleMode, Color)> = vec![
        ("Circle", KeypointStyleMode::Circle, Color::red()),
        ("Star", KeypointStyleMode::star(), Color::green()),
        (
            "Star6",
            KeypointStyleMode::Star {
                points: 6,
                inner_ratio: 0.4,
            },
            Color::blue(),
        ),
        ("Square", KeypointStyleMode::Square, Color::magenta()),
        ("Diamond", KeypointStyleMode::Diamond, Color::cyan()),
        (
            "Triangle",
            KeypointStyleMode::Triangle,
            Color::from([255u8, 165, 0, 255]),
        ), // orange
        (
            "Cross",
            KeypointStyleMode::Cross { thickness: 8 },
            Color::from([128u8, 0, 128, 255]),
        ), // purple
        (
            "X",
            KeypointStyleMode::X { thickness: 8 },
            Color::from([0u8, 100, 0, 255]),
        ), // dark green
        (
            "RoundedSq",
            KeypointStyleMode::rounded_square(),
            Color::from([70u8, 130, 180, 255]),
        ), // steel blue
        ("Glow", KeypointStyleMode::glow(), Color::red()),
        (
            "Glow3x",
            KeypointStyleMode::glow_with(3.0),
            Color::magenta(),
        ),
    ];

    let mut keypoints = Vec::new();
    for (i, (name, mode, color)) in modes.iter().enumerate() {
        let x = 70.0 + (i % 6) as f32 * 220.0;
        let y = if i < 6 { 150.0 } else { 450.0 };

        keypoints.push(
            Keypoint::default()
                .with_xy(x, y)
                .with_id(i)
                .with_name(name)
                .with_confidence(0.95)
                .with_style(
                    KeypointStyle::default()
                        .with_mode(*mode)
                        .with_radius(40)
                        .with_fill_color(ColorSource::Custom(*color))
                        .with_outline_color(ColorSource::Custom(Color::black()))
                        .with_thickness(3)
                        .with_draw_outline(true)
                        .with_text_visible(true)
                        .show_id(false)
                        .show_name(true)
                        .show_confidence(false)
                        .with_text_style(TextStyle::default().with_font_size(22.0)),
                ),
        );
    }

    let annotator = Annotator::default();
    let result = annotator.annotate(&canvas.into(), &keypoints)?;
    save_to(&result.into(), "Keypoint", "styles")?;

    Ok(())
}

// =============================================================================
// Keypoint Skeleton Demo - Large pose
// =============================================================================
fn demo_keypoint_skeleton() -> anyhow::Result<()> {
    println!("\n=== Keypoint Skeleton Demo ===");

    // Large canvas: 800x1000
    let canvas = blank_canvas(800, 1000, Color::from([250u8, 250, 250, 255]));

    let pose_kps = create_pose_keypoints_centered(400.0, 500.0, 1.2);

    // Create skeleton with colors for visibility
    let skeleton = Skeleton::from((SKELETON_COCO_19, SKELETON_COLOR_COCO_19));

    let annotator = Annotator::default().with_keypoint_style(
        KeypointStyle::default()
            .with_skeleton(skeleton)
            .with_skeleton_thickness(3)
            .with_radius(8)
            .with_fill_color(ColorSource::Custom(Color::red()))
            .with_text_visible(true)
            .show_id(true)
            .show_name(false)
            .show_confidence(false)
            .with_text_style(
                TextStyle::default()
                    .with_font_size(15.0)
                    .with_mode(TextStyleMode::rounded(3.0, 3.0)) // Rounded corners for text box
                    .with_bg_fill_color(ColorSource::AutoAlpha(200))
                    .with_bg_outline_color(ColorSource::Custom(Color::black()))
                    .with_draw_outline(true)
                    .with_thickness(1),
            ),
    );

    let result = annotator.annotate(&canvas.into(), &pose_kps)?;
    save_to(&result.into(), "Keypoint", "skeleton")?;

    Ok(())
}

fn create_pose_keypoints_centered(cx: f32, cy: f32, scale: f32) -> Vec<Keypoint> {
    let base_points = [
        (0.0, -200.0),   // 0: nose
        (15.0, -215.0),  // 1: left_eye
        (-15.0, -215.0), // 2: right_eye
        (35.0, -205.0),  // 3: left_ear
        (-35.0, -205.0), // 4: right_ear
        (60.0, -120.0),  // 5: left_shoulder
        (-60.0, -120.0), // 6: right_shoulder
        (100.0, 0.0),    // 7: left_elbow
        (-100.0, 0.0),   // 8: right_elbow
        (80.0, 100.0),   // 9: left_wrist
        (-80.0, 100.0),  // 10: right_wrist
        (40.0, 100.0),   // 11: left_hip
        (-40.0, 100.0),  // 12: right_hip
        (50.0, 230.0),   // 13: left_knee
        (-50.0, 230.0),  // 14: right_knee
        (55.0, 360.0),   // 15: left_ankle
        (-55.0, 360.0),  // 16: right_ankle
    ];

    base_points
        .iter()
        .enumerate()
        .map(|(i, (dx, dy))| {
            Keypoint::default()
                .with_xy(cx + dx * scale, cy + dy * scale)
                .with_id(i)
                .with_confidence(0.95)
        })
        .collect()
}

// =============================================================================
// Polygon Styles Demo - Showcase various polygon shapes and styles
// =============================================================================
fn demo_polygon_styles() -> anyhow::Result<()> {
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

// =============================================================================
// Prob Styles Demo - All positions on single canvas with position names
// =============================================================================
fn demo_prob_styles() -> anyhow::Result<()> {
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

// =============================================================================
// Mask Styles Demo
// =============================================================================
fn demo_mask_styles() -> anyhow::Result<()> {
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

// =============================================================================
// Combined Y Demo - On real image
// =============================================================================
fn demo_combined_y(image: &usls::Image) -> anyhow::Result<()> {
    println!("\n=== Combined Y Demo ===");

    let hbbs = vec![
        Hbb::default()
            .with_xyxy(20.0, 230.0, 795.0, 751.0)
            .with_id(5)
            .with_name("bus")
            .with_confidence(0.88),
        Hbb::default()
            .with_xyxy(669.0, 395.0, 809.0, 879.0)
            .with_id(0)
            .with_name("person")
            .with_confidence(0.87),
        Hbb::default()
            .with_xyxy(48.0, 399.0, 245.0, 903.0)
            .with_id(0)
            .with_name("person")
            .with_confidence(0.86),
    ];

    let keypoints = create_pose_keypoints();

    let probs = vec![
        Prob::default()
            .with_id(654)
            .with_name("minibus")
            .with_confidence(0.67),
        Prob::default()
            .with_id(734)
            .with_name("police_van")
            .with_confidence(0.20),
    ];

    let polygon = Polygon::try_from(vec![
        [0.0, 251.0],
        [0.0, 325.0],
        [33.0, 300.0],
        [33.0, 280.0],
        [13.0, 251.0],
    ])?
    .with_id(11)
    .with_name("stop sign")
    .with_confidence(0.56);

    let y = Y::default()
        .with_probs(&probs)
        .with_hbbs(&hbbs)
        .with_keypoints(&keypoints)
        .with_polygons(&[polygon]);

    let annotator = Annotator::default()
        .with_hbb_style(
            HbbStyle::default()
                .with_thickness(4)
                .with_draw_fill(true)
                .with_fill_color(ColorSource::AutoAlpha(50)),
        )
        .with_keypoint_style(
            KeypointStyle::default()
                .with_skeleton(SKELETON_COCO_19.into())
                .with_radius(4)
                .show_id(true)
                .show_name(false),
        )
        .with_polygon_style(PolygonStyle::default().with_text_visible(true))
        .with_prob_style(ProbStyle::default());

    let result = annotator.annotate(image, &y)?;
    let path = usls::Dir::Current
        .base_dir_with_subs(&["runs", "Annotate", "Y"])?
        .join("combined.jpg");
    result.save(path.display().to_string())?;
    println!("  Saved: Y/combined.jpg");

    Ok(())
}

fn create_pose_keypoints() -> Vec<Keypoint> {
    vec![
        Keypoint::default()
            .with_xy(139.0, 443.0)
            .with_id(0)
            .with_confidence(0.97),
        Keypoint::default()
            .with_xy(147.0, 434.0)
            .with_id(1)
            .with_confidence(0.91),
        Keypoint::default()
            .with_xy(129.0, 434.0)
            .with_id(2)
            .with_confidence(0.93),
        Keypoint::default()
            .with_xy(153.0, 442.0)
            .with_id(3)
            .with_confidence(0.60),
        Keypoint::default()
            .with_xy(106.0, 441.0)
            .with_id(4)
            .with_confidence(0.73),
        Keypoint::default()
            .with_xy(167.0, 498.0)
            .with_id(5)
            .with_confidence(0.99),
        Keypoint::default()
            .with_xy(89.0, 498.0)
            .with_id(6)
            .with_confidence(0.99),
        Keypoint::default()
            .with_xy(191.0, 575.0)
            .with_id(7)
            .with_confidence(0.95),
        Keypoint::default()
            .with_xy(116.0, 571.0)
            .with_id(8)
            .with_confidence(0.96),
        Keypoint::default()
            .with_xy(140.0, 576.0)
            .with_id(9)
            .with_confidence(0.93),
        Keypoint::default()
            .with_xy(175.0, 558.0)
            .with_id(10)
            .with_confidence(0.94),
        Keypoint::default()
            .with_xy(159.0, 652.0)
            .with_id(11)
            .with_confidence(0.98),
        Keypoint::default()
            .with_xy(99.0, 653.0)
            .with_id(12)
            .with_confidence(0.99),
        Keypoint::default()
            .with_xy(181.0, 760.0)
            .with_id(13)
            .with_confidence(0.95),
        Keypoint::default()
            .with_xy(87.0, 763.0)
            .with_id(14)
            .with_confidence(0.95),
        Keypoint::default()
            .with_xy(194.0, 861.0)
            .with_id(15)
            .with_confidence(0.80),
        Keypoint::default()
            .with_xy(71.0, 863.0)
            .with_id(16)
            .with_confidence(0.80),
    ]
}
