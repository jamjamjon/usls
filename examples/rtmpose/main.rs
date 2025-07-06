use anyhow::Result;
use usls::{
    models::{RTMPose, YOLO},
    Annotator, Config, DataLoader, Scale, Style, SKELETON_COCO_19, SKELETON_COLOR_COCO_19,
    SKELETON_COLOR_HALPE_27, SKELETON_HALPE_27,
};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// source: image, image folder, video stream
    #[argh(option, default = "String::from(\"./assets/bus.jpg\")")]
    source: String,

    /// device: cuda:0, cpu:0, ...
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale: t, s, m, l, x
    #[argh(option, default = "String::from(\"t\")")]
    scale: String,

    /// dtype: fp16, q8, q4, q4f16, ...
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// is coco 17 keypoints or halpe 26 keypoints
    #[argh(option, default = "true")]
    is_coco: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build YOLOv8
    let yolo_config = Config::yolo_detect()
        .with_scale(Scale::N)
        .with_version(8.into())
        .with_model_device(args.device.parse()?)
        .retain_classes(&[0]) // keep person class only
        .with_class_confs(&[0.5])
        .commit()?;
    let mut yolo = YOLO::new(yolo_config)?;

    // build RTMPose
    let config = match args.scale.as_str() {
        "t" => match args.is_coco {
            true => Config::rtmpose_17_t(),
            false => Config::rtmpose_26_t(),
        },
        "s" => match args.is_coco {
            true => Config::rtmpose_17_s(),
            false => Config::rtmpose_26_s(),
        },
        "m" => match args.is_coco {
            true => Config::rtmpose_17_m(),
            false => Config::rtmpose_26_m(),
        },
        "l" => match args.is_coco {
            true => Config::rtmpose_17_l(),
            false => Config::rtmpose_26_l(),
        },
        "x" => match args.is_coco {
            true => Config::rtmpose_17_x(),
            false => Config::rtmpose_26_x(),
        },
        _ => todo!(),
    }
    .with_model_dtype(args.dtype.parse()?)
    .with_model_device(args.device.parse()?)
    .commit()?;
    let mut rtmpose = RTMPose::new(config)?;

    // build annotator
    let annotator = Annotator::default()
        .with_hbb_style(Style::hbb().with_draw_fill(true))
        .with_keypoint_style(
            Style::keypoint()
                .with_radius(4)
                .with_skeleton(if args.is_coco {
                    (SKELETON_COCO_19, SKELETON_COLOR_COCO_19).into()
                } else {
                    (SKELETON_HALPE_27, SKELETON_COLOR_HALPE_27).into()
                })
                .show_id(false)
                .show_confidence(false)
                .show_name(false),
        );

    // build dataloader
    let dl = DataLoader::new(&args.source)?.with_batch(1).build()?;

    // iterate
    for xs in &dl {
        // YOLO infer
        let ys_det = yolo.forward(&xs)?;

        // RTMPose infer
        for (x, y_det) in xs.iter().zip(ys_det.iter()) {
            let y = rtmpose.forward(x, y_det.hbbs())?;

            // Annotate
            annotator.annotate(x, &y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs", rtmpose.spec()])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?
        }
    }

    usls::perf(false);

    Ok(())
}
