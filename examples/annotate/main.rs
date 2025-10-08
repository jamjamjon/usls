use usls::{Annotator, DataLoader, Hbb, Keypoint, Polygon, Prob, Style, SKELETON_COCO_19, Y};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // load images
    let image = DataLoader::try_read_one("./assets/bus.jpg")?;
    println!("Read 1 images: {:?}.", image.dimensions());

    let hbbs = vec![
        Hbb::default()
            .with_xyxy(20.81192, 229.65482, 795.1383, 751.0504)
            .with_id(5)
            .with_name("bus")
            .with_confidence(0.8815875)
            .with_style(
                // individual setting
                Style::hbb()
                    .with_thickness(5)
                    .with_draw_fill(true)
                    .with_visible(true)
                    .with_text_visible(true)
                    .show_confidence(true)
                    .show_id(true)
                    .show_name(true)
                    .with_text_loc(usls::TextLoc::Center)
                    .with_color(
                        usls::StyleColors::default()
                            .with_outline(usls::Color::white())
                            .with_fill(usls::Color::black().with_alpha(100))
                            .with_text(usls::Color::black())
                            .with_text_bg(usls::Color::white()),
                    ),
            ),
        Hbb::default()
            .with_xyxy(669.5233, 395.4491, 809.0367, 878.81226)
            .with_id(0)
            .with_name("person")
            .with_confidence(0.87094545),
        Hbb::default()
            .with_xyxy(48.03354, 398.6103, 245.06848, 902.5964)
            .with_id(0)
            .with_name("person")
            .with_confidence(0.8625425),
        Hbb::default()
            .with_xyxy(221.26727, 405.51895, 345.14288, 857.61865)
            .with_id(0)
            .with_name("person")
            .with_confidence(0.81437635),
        Hbb::default()
            .with_xyxy(0.08129883, 254.67389, 32.30627, 324.9663)
            .with_id(11)
            .with_name("stop sign")
            .with_confidence(0.30021638),
    ];

    let keypoints: Vec<Keypoint> = vec![
        Keypoint::default()
            .with_xy(139.35767, 443.43655)
            .with_id(0)
            .with_name("nose")
            .with_confidence(0.9739332),
        Keypoint::default()
            .with_xy(147.38545, 434.34055)
            .with_id(1)
            .with_name("left_eye")
            .with_confidence(0.9098319),
        Keypoint::default()
            .with_xy(128.5701, 434.07516)
            .with_id(2)
            .with_name("right_eye")
            .with_confidence(0.9320564),
        Keypoint::default()
            .with_xy(153.24237, 442.4857)
            .with_id(3)
            .with_name("left_ear")
            .with_confidence(0.5992247),
        Keypoint::default()
            .with_xy(105.74312, 441.05765)
            .with_id(4)
            .with_name("right_ear")
            .with_confidence(0.7259705),
        Keypoint::default()
            .with_xy(166.55661, 498.17484)
            .with_id(5)
            .with_name("left_shoulder")
            .with_confidence(0.9862031),
        Keypoint::default()
            .with_xy(89.40589, 497.6169)
            .with_id(6)
            .with_name("right_shoulder")
            .with_confidence(0.9879458),
        Keypoint::default()
            .with_xy(190.7351, 575.00226)
            .with_id(7)
            .with_name("left_elbow")
            .with_confidence(0.9521556),
        Keypoint::default()
            .with_xy(116.3187, 570.6441)
            .with_id(8)
            .with_name("right_elbow")
            .with_confidence(0.9619827),
        Keypoint::default()
            .with_xy(140.43465, 575.80994)
            .with_id(9)
            .with_name("left_wrist")
            .with_confidence(0.9329945),
        Keypoint::default()
            .with_xy(174.73381, 558.4027)
            .with_id(10)
            .with_name("right_wrist")
            .with_confidence(0.93989426),
        Keypoint::default()
            .with_xy(159.16801, 652.35846)
            .with_id(11)
            .with_name("left_hip")
            .with_confidence(0.9849887),
        Keypoint::default()
            .with_xy(99.27675, 653.01874)
            .with_id(12)
            .with_name("right_hip")
            .with_confidence(0.9861814),
        Keypoint::default()
            .with_xy(180.95883, 759.8797)
            .with_id(13)
            .with_name("left_knee")
            .with_confidence(0.95086014),
        Keypoint::default()
            .with_xy(87.09352, 762.6029)
            .with_id(14)
            .with_name("right_knee")
            .with_confidence(0.9532267),
        Keypoint::default()
            .with_xy(194.39137, 860.7901)
            .with_id(15)
            .with_name("left_ankle")
            .with_confidence(0.7986185),
        Keypoint::default()
            .with_xy(70.85685, 862.53253)
            .with_id(16)
            .with_name("right_ankle")
            .with_confidence(0.79832363),
    ];

    let probs = vec![
        Prob::default()
            .with_id(654)
            .with_name("minibus")
            .with_confidence(0.666985),
        Prob::default()
            .with_id(734)
            .with_name("police_van")
            .with_confidence(0.20067203),
        Prob::default()
            .with_id(874)
            .with_name("trolleybus")
            .with_confidence(0.024672432),
        Prob::default()
            .with_id(656)
            .with_name("minivan")
            .with_confidence(0.02395765),
        Prob::default()
            .with_id(757)
            .with_name("recreational_vehicle")
            .with_confidence(0.012205753),
    ];

    let polygons = vec![
        Polygon::try_from(vec![
            [13.0, 251.0],
            [12.0, 251.0],
            [11.0, 251.0],
            [10.0, 251.0],
            [9.0, 251.0],
            [8.0, 251.0],
            [7.0, 251.0],
            [6.0, 251.0],
            [5.0, 251.0],
            [4.0, 251.0],
            [3.0, 251.0],
            [2.0, 251.0],
            [1.0, 251.0],
            [0.0, 251.0],
            [0.0, 252.0],
            [0.0, 253.0],
            [0.0, 254.0],
            [0.0, 255.0],
            [0.0, 256.0],
            [0.0, 257.0],
            [0.0, 258.0],
            [0.0, 259.0],
            [0.0, 260.0],
            [0.0, 261.0],
            [0.0, 262.0],
            [0.0, 263.0],
            [0.0, 264.0],
            [0.0, 265.0],
            [0.0, 266.0],
            [0.0, 267.0],
            [0.0, 268.0],
            [0.0, 269.0],
            [0.0, 270.0],
            [0.0, 271.0],
            [0.0, 272.0],
            [0.0, 273.0],
            [0.0, 274.0],
            [0.0, 275.0],
            [0.0, 276.0],
            [0.0, 277.0],
            [0.0, 278.0],
            [0.0, 279.0],
            [0.0, 280.0],
            [0.0, 281.0],
            [0.0, 282.0],
            [0.0, 283.0],
            [0.0, 284.0],
            [0.0, 285.0],
            [0.0, 286.0],
            [0.0, 287.0],
            [0.0, 288.0],
            [0.0, 289.0],
            [0.0, 290.0],
            [0.0, 291.0],
            [0.0, 292.0],
            [0.0, 293.0],
            [0.0, 294.0],
            [0.0, 295.0],
            [0.0, 296.0],
            [0.0, 297.0],
            [0.0, 298.0],
            [0.0, 299.0],
            [0.0, 300.0],
            [0.0, 301.0],
            [0.0, 302.0],
            [0.0, 303.0],
            [0.0, 304.0],
            [0.0, 305.0],
            [0.0, 306.0],
            [0.0, 307.0],
            [0.0, 308.0],
            [0.0, 309.0],
            [0.0, 310.0],
            [0.0, 311.0],
            [0.0, 312.0],
            [0.0, 313.0],
            [0.0, 314.0],
            [0.0, 315.0],
            [0.0, 316.0],
            [0.0, 317.0],
            [0.0, 318.0],
            [0.0, 319.0],
            [0.0, 320.0],
            [0.0, 321.0],
            [0.0, 322.0],
            [0.0, 323.0],
            [0.0, 324.0],
            [0.0, 325.0],
            [1.0, 325.0],
            [2.0, 325.0],
            [3.0, 325.0],
            [4.0, 325.0],
            [5.0, 325.0],
            [6.0, 325.0],
            [7.0, 325.0],
            [8.0, 325.0],
            [9.0, 325.0],
            [10.0, 325.0],
            [11.0, 325.0],
            [12.0, 324.0],
            [13.0, 324.0],
            [14.0, 324.0],
            [15.0, 323.0],
            [16.0, 323.0],
            [17.0, 322.0],
            [18.0, 321.0],
            [19.0, 321.0],
            [20.0, 320.0],
            [20.0, 319.0],
            [21.0, 318.0],
            [22.0, 317.0],
            [23.0, 316.0],
            [24.0, 315.0],
            [24.0, 314.0],
            [25.0, 313.0],
            [26.0, 312.0],
            [27.0, 311.0],
            [28.0, 310.0],
            [29.0, 309.0],
            [30.0, 308.0],
            [30.0, 307.0],
            [31.0, 306.0],
            [31.0, 305.0],
            [31.0, 304.0],
            [32.0, 303.0],
            [32.0, 302.0],
            [32.0, 301.0],
            [33.0, 300.0],
            [33.0, 299.0],
            [33.0, 298.0],
            [33.0, 297.0],
            [33.0, 296.0],
            [33.0, 295.0],
            [33.0, 294.0],
            [33.0, 293.0],
            [33.0, 292.0],
            [33.0, 291.0],
            [33.0, 290.0],
            [33.0, 289.0],
            [33.0, 288.0],
            [33.0, 287.0],
            [33.0, 286.0],
            [33.0, 285.0],
            [33.0, 284.0],
            [33.0, 283.0],
            [33.0, 282.0],
            [33.0, 281.0],
            [33.0, 280.0],
            [32.0, 279.0],
            [32.0, 278.0],
            [32.0, 277.0],
            [31.0, 276.0],
            [31.0, 275.0],
            [31.0, 274.0],
            [30.0, 273.0],
            [30.0, 272.0],
            [29.0, 271.0],
            [28.0, 270.0],
            [28.0, 269.0],
            [27.0, 268.0],
            [27.0, 267.0],
            [26.0, 266.0],
            [25.0, 265.0],
            [25.0, 264.0],
            [24.0, 263.0],
            [23.0, 262.0],
            [22.0, 261.0],
            [21.0, 260.0],
            [20.0, 259.0],
            [20.0, 258.0],
            [19.0, 257.0],
            [18.0, 256.0],
            [17.0, 255.0],
            [16.0, 254.0],
            [15.0, 254.0],
            [14.0, 253.0],
            [13.0, 252.0],
            [13.0, 251.0],
        ])?
        .with_id(11)
        .with_name("stop sign")
        .with_confidence(0.5555),
        Polygon::try_from(vec![
            [485.0, 149.0],
            [484.0, 150.0],
            [484.0, 151.0],
            [483.0, 152.0],
            [482.0, 153.0],
            [481.0, 153.0],
            [480.0, 153.0],
            [479.0, 153.0],
            [478.0, 153.0],
            [477.0, 154.0],
            [476.0, 154.0],
            [475.0, 154.0],
            [474.0, 154.0],
            [473.0, 154.0],
            [472.0, 154.0],
            [471.0, 154.0],
            [470.0, 154.0],
            [469.0, 154.0],
            [468.0, 155.0],
            [467.0, 155.0],
            [466.0, 155.0],
            [465.0, 155.0],
            [464.0, 155.0],
            [463.0, 155.0],
            [462.0, 156.0],
            [461.0, 156.0],
            [460.0, 156.0],
            [459.0, 156.0],
            [458.0, 156.0],
            [457.0, 157.0],
            [456.0, 157.0],
            [455.0, 157.0],
            [454.0, 157.0],
            [453.0, 158.0],
            [452.0, 158.0],
            [451.0, 158.0],
            [450.0, 158.0],
            [449.0, 159.0],
            [448.0, 159.0],
            [447.0, 159.0],
            [446.0, 159.0],
            [445.0, 160.0],
            [444.0, 160.0],
            [443.0, 160.0],
            [442.0, 160.0],
            [441.0, 160.0],
            [440.0, 161.0],
            [439.0, 161.0],
            [438.0, 161.0],
            [437.0, 161.0],
            [436.0, 161.0],
            [435.0, 162.0],
            [434.0, 162.0],
            [433.0, 162.0],
            [432.0, 162.0],
            [431.0, 162.0],
            [430.0, 162.0],
            [429.0, 163.0],
            [428.0, 163.0],
            [427.0, 163.0],
            [427.0, 164.0],
            [427.0, 165.0],
            [427.0, 166.0],
            [427.0, 167.0],
            [427.0, 168.0],
            [427.0, 169.0],
            [427.0, 170.0],
            [427.0, 171.0],
            [427.0, 172.0],
            [427.0, 173.0],
            [427.0, 174.0],
            [427.0, 175.0],
            [427.0, 176.0],
            [427.0, 177.0],
            [427.0, 178.0],
            [427.0, 179.0],
            [427.0, 180.0],
            [427.0, 181.0],
            [427.0, 182.0],
            [427.0, 183.0],
            [427.0, 184.0],
            [427.0, 185.0],
            [427.0, 186.0],
            [427.0, 187.0],
            [427.0, 188.0],
            [427.0, 189.0],
            [427.0, 190.0],
            [428.0, 190.0],
            [429.0, 191.0],
            [430.0, 191.0],
            [431.0, 191.0],
            [432.0, 191.0],
            [433.0, 191.0],
            [434.0, 191.0],
            [435.0, 191.0],
            [436.0, 191.0],
            [437.0, 191.0],
            [438.0, 190.0],
            [439.0, 190.0],
            [440.0, 190.0],
            [441.0, 190.0],
            [442.0, 190.0],
            [443.0, 190.0],
            [444.0, 190.0],
            [445.0, 189.0],
            [446.0, 189.0],
            [447.0, 189.0],
            [448.0, 189.0],
            [449.0, 189.0],
            [450.0, 189.0],
            [451.0, 188.0],
            [452.0, 188.0],
            [453.0, 188.0],
            [454.0, 188.0],
            [455.0, 188.0],
            [456.0, 188.0],
            [457.0, 187.0],
            [458.0, 187.0],
            [459.0, 187.0],
            [460.0, 187.0],
            [461.0, 186.0],
            [462.0, 186.0],
            [463.0, 187.0],
            [464.0, 188.0],
            [465.0, 189.0],
            [466.0, 190.0],
            [467.0, 191.0],
            [467.0, 192.0],
            [468.0, 193.0],
            [469.0, 193.0],
            [470.0, 193.0],
            [471.0, 193.0],
            [472.0, 193.0],
            [473.0, 193.0],
            [474.0, 193.0],
            [475.0, 193.0],
            [476.0, 193.0],
            [477.0, 193.0],
            [478.0, 192.0],
            [479.0, 191.0],
            [480.0, 190.0],
            [481.0, 190.0],
            [482.0, 189.0],
            [483.0, 189.0],
            [484.0, 189.0],
            [485.0, 188.0],
            [486.0, 188.0],
            [487.0, 188.0],
            [488.0, 188.0],
            [489.0, 188.0],
            [490.0, 188.0],
            [491.0, 188.0],
            [492.0, 188.0],
            [493.0, 187.0],
            [494.0, 187.0],
            [495.0, 187.0],
            [496.0, 187.0],
            [497.0, 187.0],
            [498.0, 187.0],
            [499.0, 187.0],
            [500.0, 186.0],
            [501.0, 186.0],
            [502.0, 186.0],
            [503.0, 186.0],
            [504.0, 185.0],
            [505.0, 185.0],
            [506.0, 185.0],
            [507.0, 184.0],
            [508.0, 184.0],
            [509.0, 183.0],
            [510.0, 183.0],
            [511.0, 183.0],
            [512.0, 182.0],
            [513.0, 182.0],
            [514.0, 182.0],
            [515.0, 181.0],
            [516.0, 181.0],
            [517.0, 181.0],
            [518.0, 180.0],
            [519.0, 180.0],
            [520.0, 180.0],
            [521.0, 179.0],
            [522.0, 179.0],
            [523.0, 178.0],
            [524.0, 178.0],
            [525.0, 177.0],
            [526.0, 176.0],
            [527.0, 175.0],
            [528.0, 174.0],
            [529.0, 173.0],
            [530.0, 172.0],
            [531.0, 172.0],
            [531.0, 171.0],
            [531.0, 170.0],
            [531.0, 169.0],
            [531.0, 168.0],
            [531.0, 167.0],
            [531.0, 166.0],
            [531.0, 165.0],
            [531.0, 164.0],
            [531.0, 163.0],
            [531.0, 162.0],
            [531.0, 161.0],
            [531.0, 160.0],
            [531.0, 159.0],
            [531.0, 158.0],
            [531.0, 157.0],
            [531.0, 156.0],
            [530.0, 155.0],
            [530.0, 154.0],
            [529.0, 154.0],
            [528.0, 153.0],
            [527.0, 152.0],
            [526.0, 151.0],
            [525.0, 150.0],
            [524.0, 149.0],
            [523.0, 149.0],
            [522.0, 149.0],
            [521.0, 149.0],
            [520.0, 149.0],
            [519.0, 149.0],
            [518.0, 149.0],
            [517.0, 149.0],
            [516.0, 149.0],
            [515.0, 149.0],
            [514.0, 149.0],
            [513.0, 149.0],
            [512.0, 149.0],
            [511.0, 149.0],
            [510.0, 149.0],
            [509.0, 149.0],
            [508.0, 149.0],
            [507.0, 149.0],
            [506.0, 149.0],
            [505.0, 149.0],
            [504.0, 149.0],
            [503.0, 149.0],
            [502.0, 149.0],
            [501.0, 149.0],
            [500.0, 149.0],
            [499.0, 149.0],
            [498.0, 149.0],
            [497.0, 149.0],
            [496.0, 149.0],
            [495.0, 149.0],
            [494.0, 149.0],
            [493.0, 149.0],
            [492.0, 149.0],
            [491.0, 149.0],
            [490.0, 149.0],
            [489.0, 149.0],
            [488.0, 149.0],
            [487.0, 149.0],
            [486.0, 149.0],
            [485.0, 149.0],
        ])?
        .with_id(9)
        .with_name("traffic light")
        .with_confidence(0.777777),
    ];

    // Build annotator
    let annotator = Annotator::default()
        .with_prob_style(Style::prob().with_text_loc(usls::TextLoc::InnerTopLeft))
        .with_hbb_style(Style::hbb().with_thickness(5).with_draw_fill(true))
        .with_keypoint_style(
            Style::keypoint()
                .with_skeleton(SKELETON_COCO_19.into())
                .with_radius(4)
                .with_text_visible(true)
                .show_confidence(false)
                .show_id(true)
                .show_name(false),
        )
        .with_polygon_style(
            Style::polygon()
                .with_text_visible(true)
                .show_confidence(true)
                .show_id(true)
                .show_name(true),
        );

    // Annotate Y
    let y = Y::default()
        .with_probs(&probs)
        .with_hbbs(&hbbs)
        .with_keypoints(&keypoints)
        // .with_keypointss(&[keypoints.clone()])
        .with_polygons(&polygons);
    annotator.annotate(&image, &y)?.save(format!(
        "{}.jpg",
        usls::Dir::Current
            .base_dir_with_subs(&["runs", "Annotate", "Y"])?
            .join(usls::timestamp(None))
            .display(),
    ))?;

    // Annotate Probs
    annotator.annotate(&image, &probs)?.save(format!(
        "{}.jpg",
        usls::Dir::Current
            .base_dir_with_subs(&["runs", "Annotate", "Probs"])?
            .join(usls::timestamp(None))
            .display(),
    ))?;

    // Annotate Prob
    for prob in &probs {
        annotator.annotate(&image, prob)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", "Annotate", "Prob"])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    // Annotate Hbbs
    annotator.annotate(&image, &hbbs)?.save(format!(
        "{}.jpg",
        usls::Dir::Current
            .base_dir_with_subs(&["runs", "Annotate", "Hbbs"])?
            .join(usls::timestamp(None))
            .display(),
    ))?;

    // Annotate Hbb
    for hbb in &hbbs {
        annotator.annotate(&image, hbb)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", "Annotate", "Hbb"])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    // Annotate A set of Keypoint
    annotator.annotate(&image, &keypoints)?.save(format!(
        "{}.jpg",
        usls::Dir::Current
            .base_dir_with_subs(&["runs", "Annotate", "Keypoints"])?
            .join(usls::timestamp(None))
            .display(),
    ))?;

    // Annotate Keypoint
    for keypoint in &keypoints {
        annotator.annotate(&image, keypoint)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", "Annotate", "Keypoint"])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    // Annotate Polygons
    annotator.annotate(&image, &polygons)?.save(format!(
        "{}.jpg",
        usls::Dir::Current
            .base_dir_with_subs(&["runs", "Annotate", "Polygons"])?
            .join(usls::timestamp(None))
            .display(),
    ))?;

    // Annotate Polygon
    for polygon in &polygons {
        annotator.annotate(&image, polygon)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", "Annotate", "Polygon"])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    Ok(())
}
