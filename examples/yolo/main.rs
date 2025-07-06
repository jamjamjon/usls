use anyhow::Result;
use usls::{
    models::YOLO, Annotator, Config, DataLoader, Style, NAMES_COCO_80, NAMES_COCO_KEYPOINTS_17,
    NAMES_IMAGENET_1K, SKELETON_COCO_19, SKELETON_COLOR_COCO_19,
};

#[derive(argh::FromArgs, Debug)]
/// YOLO Example
struct Args {
    /// model file(.onnx)
    #[argh(option)]
    model: Option<String>,

    /// source: image, image folder, video stream
    #[argh(option, default = "String::from(\"./assets/bus.jpg\")")]
    source: String,

    /// model dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// task: det, seg, pose, classify, obb
    #[argh(option, default = "String::from(\"det\")")]
    task: String,

    /// version
    #[argh(option, default = "8.0")]
    ver: f32,

    /// device: cuda, cpu, mps
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale: n, s, m, l, x
    #[argh(option, default = "String::from(\"n\")")]
    scale: String,

    /// enable TensorRT FP16
    #[argh(option, default = "true")]
    trt_fp16: bool,

    /// batch size
    #[argh(option, default = "1")]
    batch_size: usize,

    /// bin batch size: For TensorRT
    #[argh(option, default = "1")]
    min_batch_size: usize,

    /// max Batch size: For TensorRT
    #[argh(option, default = "4")]
    max_batch_size: usize,

    /// min image width: For TensorRT
    #[argh(option, default = "224")]
    min_image_width: isize,

    /// image width: For TensorRT
    #[argh(option, default = "640")]
    image_width: isize,

    /// max image width: For TensorRT
    #[argh(option, default = "1280")]
    max_image_width: isize,

    /// min image height: For TensorRT
    #[argh(option, default = "224")]
    min_image_height: isize,

    /// image height: For TensorRT
    #[argh(option, default = "640")]
    image_height: isize,

    /// max image height: For TensorRT
    #[argh(option, default = "1280")]
    max_image_height: isize,

    /// num classes
    #[argh(option)]
    num_classes: Option<usize>,

    /// num keypoints
    #[argh(option)]
    num_keypoints: Option<usize>,

    /// class names
    #[argh(option)]
    class_names: Vec<String>,

    /// keypoint names
    #[argh(option)]
    keypoint_names: Vec<String>,

    /// top-k
    #[argh(option, default = "5")]
    topk: usize,

    /// use COCO 80 classes
    #[argh(switch)]
    use_coco_80_classes: bool,

    /// use COCO 17 keypoints classes
    #[argh(switch)]
    use_coco_17_keypoints_classes: bool,

    /// use ImageNet 1K classes
    #[argh(switch)]
    use_imagenet_1k_classes: bool,

    /// confidences
    #[argh(option)]
    confs: Vec<f32>,

    /// keypoint nonfidences
    #[argh(option)]
    keypoint_confs: Vec<f32>,

    /// exclude nlasses
    #[argh(option)]
    exclude_classes: Vec<usize>,

    /// retain classes
    #[argh(option)]
    retain_classes: Vec<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();
    let mut config = Config::yolo()
        .with_model_file(&args.model.unwrap_or_default())
        .with_task(args.task.parse()?)
        .with_version(args.ver.try_into()?)
        .with_scale(args.scale.parse()?)
        .with_model_dtype(args.dtype.parse()?)
        .with_model_device(args.device.parse()?)
        .with_model_tensorrt_fp16(args.trt_fp16)
        .with_model_ixx(
            0,
            0,
            (args.min_batch_size, args.batch_size, args.max_batch_size).into(),
        )
        .with_model_ixx(
            0,
            2,
            (
                args.min_image_height,
                args.image_height,
                args.max_image_height,
            )
                .into(),
        )
        .with_model_ixx(
            0,
            3,
            (args.min_image_width, args.image_width, args.max_image_width).into(),
        )
        .with_class_confs(if args.confs.is_empty() {
            &[0.2, 0.15]
        } else {
            &args.confs
        })
        .with_keypoint_confs(if args.keypoint_confs.is_empty() {
            &[0.5]
        } else {
            &args.keypoint_confs
        })
        .with_topk(args.topk)
        .retain_classes(&args.retain_classes)
        .exclude_classes(&args.exclude_classes)
        .with_model_num_dry_run(2);
    if args.use_coco_80_classes {
        config = config.with_class_names(&NAMES_COCO_80);
    }
    if args.use_coco_17_keypoints_classes {
        config = config.with_keypoint_names(&NAMES_COCO_KEYPOINTS_17);
    }
    if args.use_imagenet_1k_classes {
        config = config.with_class_names(&NAMES_IMAGENET_1K);
    }
    if let Some(nc) = args.num_classes {
        config = config.with_nc(nc);
    }
    if let Some(nk) = args.num_keypoints {
        config = config.with_nk(nk);
    }
    if !args.class_names.is_empty() {
        config = config.with_class_names(
            &args
                .class_names
                .iter()
                .map(|x| x.as_str())
                .collect::<Vec<_>>(),
        );
    }
    if !args.keypoint_names.is_empty() {
        config = config.with_keypoint_names(
            &args
                .keypoint_names
                .iter()
                .map(|x| x.as_str())
                .collect::<Vec<_>>(),
        );
    }

    // build model
    let mut model = YOLO::new(config.commit()?)?;

    // build dataloader
    let dl = DataLoader::new(&args.source)?
        .with_batch(model.batch() as _)
        .build()?;

    // build annotator
    let annotator = Annotator::default()
        .with_obb_style(Style::obb().with_draw_fill(true))
        .with_hbb_style(
            Style::hbb()
                .with_draw_fill(true)
                .with_palette(&usls::Color::palette_coco_80()),
        )
        .with_keypoint_style(
            Style::keypoint()
                .with_skeleton((SKELETON_COCO_19, SKELETON_COLOR_COCO_19).into())
                .show_confidence(false)
                .show_id(true)
                .show_name(false),
        )
        .with_mask_style(Style::mask().with_draw_mask_polygon_largest(true));

    // run & annotate
    for xs in &dl {
        let ys = model.forward(&xs)?;
        // println!("ys: {:?}", ys);

        for (x, y) in xs.iter().zip(ys.iter()) {
            annotator.annotate(x, y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs", model.spec()])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?;
        }
    }

    usls::perf(false);

    Ok(())
}
