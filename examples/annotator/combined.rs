use crate::keypoint::create_pose_keypoints;
use usls::{
    Annotator, ColorSource, Hbb, HbbStyle, KeypointStyle, Polygon, PolygonStyle, Prob, ProbStyle,
    SKELETON_COCO_19, Y,
};

pub fn demo_combined_y(image: &usls::Image) -> anyhow::Result<()> {
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
