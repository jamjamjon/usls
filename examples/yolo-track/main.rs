use anyhow::Result;

use usls::{
    models::YOLO, Annotator, ByteTracker, Config, DataLoader, Viewer, SKELETON_COCO_19,
    SKELETON_COLOR_COCO_19,
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
    #[argh(option, default = "String::from(\"m\")")]
    scale: String,

    /// batch size
    #[argh(option, default = "1")]
    batch_size: usize,

    /// min batch size: For TensorRT
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

    /// confidences
    #[argh(option)]
    confs: Vec<f32>,

    /// retain classes
    #[argh(option)]
    classes: Vec<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();
    let config = Config::yolo()
        .with_model_file(args.model.unwrap_or_default())
        .with_task(args.task.parse()?)
        .with_version(args.ver.try_into()?)
        .with_scale(args.scale.parse()?)
        .with_model_dtype(args.dtype.parse()?)
        .with_model_device(args.device.parse()?)
        .with_model_ixx(
            0,
            0,
            (args.min_batch_size, args.batch_size, args.max_batch_size),
        )
        .with_model_ixx(
            0,
            2,
            (
                args.min_image_height,
                args.image_height,
                args.max_image_height,
            ),
        )
        .with_model_ixx(
            0,
            3,
            (args.min_image_width, args.image_width, args.max_image_width),
        )
        .with_class_confs(if args.confs.is_empty() {
            &[0.25]
        } else {
            &args.confs
        })
        .retain_classes(&args.classes);

    // build model
    let mut model = YOLO::new(config.commit()?)?;

    // build dataloader
    let dl = DataLoader::new(&args.source)?
        .with_batch(model.batch() as _)
        .build()?;

    // build annotator
    let annotator = Annotator::default()
        .with_obb_style(usls::ObbStyle::default().with_draw_fill(true))
        .with_hbb_style(
            usls::HbbStyle::default()
                .with_draw_fill(true)
                .with_palette(&usls::Color::palette_coco_80()),
        )
        .with_keypoint_style(
            usls::KeypointStyle::default()
                .with_skeleton((SKELETON_COCO_19, SKELETON_COLOR_COCO_19).into())
                .show_confidence(false)
                .show_id(true)
                .show_name(false),
        )
        .with_mask_style(usls::MaskStyle::default().with_draw_polygon_largest(true));

    // build viewer and tracker
    let mut viewer = Viewer::default().with_window_scale(1.);
    let mut tracker = ByteTracker::default().with_max_age(60);

    // run & annotate
    for xs in &dl {
        // check out window
        if viewer.is_window_exist() && !viewer.is_window_open() {
            break;
        }

        let ys = model.forward(&xs)?;

        let hbbs = ys[0].hbbs();
        if !hbbs.is_empty() {
            let tracked_hbbs = tracker.update(hbbs)?;

            // annnotate
            let image_annotated = annotator.annotate(&xs[0], &tracked_hbbs)?;

            // imshow
            viewer.imshow(&image_annotated)?;
        } else {
            continue;
        }

        // check out key event
        if let Some(key) = viewer.wait_key(10) {
            if key == usls::Key::Escape {
                break;
            }
        }
    }

    usls::perf(false);

    Ok(())
}
