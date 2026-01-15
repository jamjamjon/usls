use anyhow::Result;
use clap::Args;
use usls::{
    Config, DType, Device, Scale, Task, NAMES_COCO_80, NAMES_COCO_KEYPOINTS_17, NAMES_IMAGENET_1K,
};

#[derive(Debug, Args)]
pub struct YoloArgs {
    /// Model file
    #[arg(long)]
    pub model: Option<String>,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, global = true, default_value = "fp16")]
    pub dtype: DType,

    /// Task: det, seg, pose, classify, obb
    #[arg(long, global = true, default_value = "det")]
    pub task: Task,

    /// Version
    #[arg(long, global = true, default_value_t = 8.0)]
    pub ver: f32,

    /// Scale: n, s, m, l, x
    #[arg(long, global = true, default_value = "n")]
    pub scale: Scale,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

    /// Batch size
    #[arg(long, global = true, default_value_t = 1)]
    pub batch: usize,

    /// Min batch size (TensorRT)
    #[arg(long, global = true, default_value_t = 1)]
    pub min_batch: usize,

    /// Max batch size (TensorRT)
    #[arg(long, global = true, default_value_t = 4)]
    pub max_batch: usize,

    /// Image width
    #[arg(long, global = true, default_value_t = 640)]
    pub image_width: isize,

    /// Image height
    #[arg(long, global = true, default_value_t = 640)]
    pub image_height: isize,

    /// Min image width (TensorRT)
    #[arg(long, global = true, default_value_t = 224)]
    pub min_image_width: isize,

    /// Max image width (TensorRT)
    #[arg(long, global = true, default_value_t = 1280)]
    pub max_image_width: isize,

    /// Min image height (TensorRT)
    #[arg(long, global = true, default_value_t = 224)]
    pub min_image_height: isize,

    /// Max image height (TensorRT)
    #[arg(long, global = true, default_value_t = 1280)]
    pub max_image_height: isize,

    /// num dry run
    #[arg(long, global = true, default_value_t = 5)]
    pub num_dry_run: usize,

    /// Number of classes
    #[arg(long)]
    pub num_classes: Option<usize>,

    /// Number of keypoints
    #[arg(long)]
    pub num_keypoints: Option<usize>,

    /// Class names
    #[arg(long, value_delimiter = ',')]
    pub class_names: Vec<String>,

    /// Keypoint names
    #[arg(long, value_delimiter = ',')]
    pub keypoint_names: Vec<String>,

    /// Use COCO 80 classes
    #[arg(long, action)]
    pub use_coco_80_classes: bool,

    /// Use COCO 17 keypoints classes
    #[arg(long, action)]
    pub use_coco_17_keypoints_classes: bool,

    /// Use ImageNet 1K classes
    #[arg(long, action)]
    pub use_imagenet_1k_classes: bool,
}

pub fn config(args: &YoloArgs) -> Result<Config> {
    let mut config = Config::yolo();
    if let Some(model) = &args.model {
        config = config.with_model_file(model);
    }
    config = config
        .with_task(args.task.clone())
        .with_version(args.ver.try_into()?)
        .with_scale(args.scale.clone())
        .with_model_dtype(args.dtype)
        .with_model_device(args.device)
        .with_model_ixx(0, 0, (args.min_batch, args.batch, args.max_batch))
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
        .with_image_processor_device(args.processor_device)
        .with_model_num_dry_run(args.num_dry_run);

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
        config = config.with_class_names_owned(args.class_names.clone());
    }
    if !args.keypoint_names.is_empty() {
        config = config.with_keypoint_names_owned(args.keypoint_names.clone());
    }

    Ok(config)
}
