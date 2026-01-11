use clap::Parser;
use image::RgbaImage;
use usls::{
    Annotator, Color, ColorSource, Keypoint, KeypointStyle, KeypointStyleMode, Skeleton, TextStyle,
    TextStyleMode, SKELETON_COCO_19, SKELETON_COLOR_COCO_19,
};

#[derive(Parser, Debug)]
pub struct KeypointArgs {}

pub fn run(
    canvas_fn: impl Fn(u32, u32, Color) -> RgbaImage,
    save_fn: impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    demo_keypoint_styles(&canvas_fn, &save_fn)?;
    demo_keypoint_skeleton(&canvas_fn, &save_fn)?;
    Ok(())
}

fn demo_keypoint_styles(
    blank_canvas: &impl Fn(u32, u32, Color) -> RgbaImage,
    save_to: &impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
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

fn demo_keypoint_skeleton(
    blank_canvas: &impl Fn(u32, u32, Color) -> RgbaImage,
    save_to: &impl Fn(&RgbaImage, &str, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
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

pub fn create_pose_keypoints() -> Vec<Keypoint> {
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
