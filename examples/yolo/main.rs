use anyhow::Result;
use clap::Parser;

use usls::{
    models::YOLO, Annotator, DataLoader, Options, Vision, YOLOTask, YOLOVersion, COCO_KEYPOINTS_17,
    COCO_SKELETONS_16,
};

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long)]
    pub model: Option<String>,

    #[arg(long, default_value_t = String::from("./assets/bus.jpg"))]
    pub source: String,

    #[arg(long, value_enum, default_value_t = YOLOTask::Detect)]
    pub task: YOLOTask,

    #[arg(long, value_enum, default_value_t = YOLOVersion::V8)]
    pub ver: YOLOVersion,

    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 224)]
    pub width_min: isize,

    #[arg(long, default_value_t = 640)]
    pub width: isize,

    #[arg(long, default_value_t = 800)]
    pub width_max: isize,

    #[arg(long, default_value_t = 224)]
    pub height_min: isize,

    #[arg(long, default_value_t = 640)]
    pub height: isize,

    #[arg(long, default_value_t = 800)]
    pub height_max: isize,

    #[arg(long, default_value_t = 80)]
    pub nc: usize,

    #[arg(long)]
    pub trt: bool,

    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub half: bool,

    #[arg(long)]
    pub coreml: bool,

    #[arg(long, default_value_t = 0)]
    pub device_id: usize,

    #[arg(long)]
    pub profile: bool,

    #[arg(long)]
    pub no_plot: bool,

    #[arg(long)]
    pub no_contours: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // build options
    let options = Options::default();

    // version & task
    let (options, saveout) = match args.ver {
        YOLOVersion::V5 => match args.task {
            YOLOTask::Classify => (
                options.with_model(&args.model.unwrap_or("yolo/v5-n-cls-dyn.onnx".to_string()))?,
                "YOLOv5-Classify",
            ),
            YOLOTask::Detect => (
                options.with_model(&args.model.unwrap_or("yolo/v5-n-dyn.onnx".to_string()))?,
                "YOLOv5-Detect",
            ),
            YOLOTask::Segment => (
                options.with_model(&args.model.unwrap_or("yolo/v5-n-seg-dyn.onnx".to_string()))?,
                "YOLOv5-Segment",
            ),
            t => anyhow::bail!("Task: {t:?} is unsupported for {:?}", args.ver),
        },
        YOLOVersion::V6 => match args.task {
            YOLOTask::Detect => (
                options
                    .with_model(&args.model.unwrap_or("yolo/v6-n-dyn.onnx".to_string()))?
                    .with_nc(args.nc),
                "YOLOv6-Detect",
            ),
            t => anyhow::bail!("Task: {t:?} is unsupported for {:?}", args.ver),
        },
        YOLOVersion::V7 => match args.task {
            YOLOTask::Detect => (
                options
                    .with_model(&args.model.unwrap_or("yolo/v7-tiny-dyn.onnx".to_string()))?
                    .with_nc(args.nc),
                "YOLOv7-Detect",
            ),
            t => anyhow::bail!("Task: {t:?} is unsupported for {:?}", args.ver),
        },
        YOLOVersion::V8 => match args.task {
            YOLOTask::Classify => (
                options.with_model(&args.model.unwrap_or("yolo/v8-m-cls-dyn.onnx".to_string()))?,
                "YOLOv8-Classify",
            ),
            YOLOTask::Detect => (
                options.with_model(&args.model.unwrap_or("yolo/v8-m-dyn.onnx".to_string()))?,
                "YOLOv8-Detect",
            ),
            YOLOTask::Segment => (
                options.with_model(&args.model.unwrap_or("yolo/v8-m-seg-dyn.onnx".to_string()))?,
                "YOLOv8-Segment",
            ),
            YOLOTask::Pose => (
                options.with_model(&args.model.unwrap_or("yolo/v8-m-pose-dyn.onnx".to_string()))?,
                "YOLOv8-Pose",
            ),
            YOLOTask::Obb => (
                options.with_model(&args.model.unwrap_or("yolo/v8-m-obb-dyn.onnx".to_string()))?,
                "YOLOv8-Obb",
            ),
        },
        YOLOVersion::V9 => match args.task {
            YOLOTask::Detect => (
                options.with_model(&args.model.unwrap_or("yolo/v9-c-dyn-f16.onnx".to_string()))?,
                "YOLOv9-Detect",
            ),
            t => anyhow::bail!("Task: {t:?} is unsupported for {:?}", args.ver),
        },
        YOLOVersion::V10 => match args.task {
            YOLOTask::Detect => (
                options.with_model(&args.model.unwrap_or("yolo/v10-n.onnx".to_string()))?,
                "YOLOv10-Detect",
            ),
            t => anyhow::bail!("Task: {t:?} is unsupported for {:?}", args.ver),
        },
        YOLOVersion::RTDETR => match args.task {
            YOLOTask::Detect => (
                options.with_model(&args.model.unwrap_or("yolo/rtdetr-l-f16.onnx".to_string()))?,
                "RTDETR",
            ),
            t => anyhow::bail!("Task: {t:?} is unsupported for {:?}", args.ver),
        },
    };

    let options = options
        .with_yolo_version(args.ver)
        .with_yolo_task(args.task);

    // device
    let options = if args.cuda {
        options.with_cuda(args.device_id)
    } else if args.trt {
        let options = options.with_trt(args.device_id);
        if args.half {
            options.with_fp16(true)
        } else {
            options
        }
    } else if args.coreml {
        options.with_coreml(args.device_id)
    } else {
        options.with_cpu()
    };
    let options = options
        .with_ixx(0, 0, (1, args.batch_size as _, 4).into())
        .with_ixx(0, 2, (args.height_min, args.height, args.height_max).into())
        .with_ixx(0, 3, (args.width_min, args.width, args.width_max).into())
        .with_confs(&[0.2, 0.15]) // class_0: 0.4, others: 0.15
        // .with_names(&COCO_CLASS_NAMES_80)
        .with_names2(&COCO_KEYPOINTS_17)
        .with_find_contours(!args.no_contours) // find contours or not
        .with_profile(args.profile);
    let mut model = YOLO::new(options)?;

    // build dataloader
    let dl = DataLoader::new(&args.source)?
        .with_batch(model.batch() as _)
        .build()?;

    // build annotator
    let annotator = Annotator::default()
        .with_skeletons(&COCO_SKELETONS_16)
        .with_bboxes_thickness(4)
        .without_masks(true) // No masks plotting when doing segment task.
        .with_saveout(saveout);

    // run & annotate
    for (xs, _paths) in dl {
        // let ys = model.run(&xs)?;  // way one
        let ys = model.forward(&xs, args.profile)?; // way two
        if !args.no_plot {
            annotator.annotate(&xs, &ys);
        }
    }

    Ok(())
}
