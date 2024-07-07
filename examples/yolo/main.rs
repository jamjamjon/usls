use clap::Parser;

use usls::{
    coco, models::YOLO, Annotator, DataLoader, Options, Vision, YOLOFormat, YOLOTask, YOLOVersion,
};

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, default_value_t = String::from("./assets/bus.jpg"))]
    pub source: String,

    #[arg(long, value_enum, default_value_t = YOLOTask::Detect)]
    pub task: YOLOTask,

    #[arg(long, value_enum, default_value_t = YOLOVersion::V8)]
    pub version: YOLOVersion,

    #[arg(long, value_enum, default_value_t = YOLOFormat::NCxcywhClssA)]
    pub format: YOLOFormat,

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
    pub plot: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // build options
    let options = Options::default();

    // version & task
    let options = match args.version {
        YOLOVersion::V5 => match args.task {
            YOLOTask::Classify => options.with_model("../models/yolov5s-cls.onnx")?,
            YOLOTask::Detect => options.with_model("../models/yolov5s.onnx")?,
            YOLOTask::Segment => options.with_model("../models/yolov5s.onnx")?,
            t => todo!("{t:?} is unsupported for {:?}", args.version),
        },
        YOLOVersion::V8 => match args.task {
            YOLOTask::Classify => options.with_model("yolov8m-cls-dyn-cls.onnx")?,
            YOLOTask::Detect => options.with_model("yolov8m-dyn.onnx")?,
            YOLOTask::Segment => options.with_model("yolov8m-seg-dyn.onnx")?,
            YOLOTask::Pose => options.with_model("yolov8m-pose-dyn.onnx")?,
            YOLOTask::Obb => options.with_model("yolov8m-obb-dyn.onnx")?,
        },
        YOLOVersion::V9 => match args.task {
            YOLOTask::Detect => options.with_model("yolov9-c-dyn-f16.onnx")?,
            t => todo!("{t:?} is unsupported for {:?}", args.version),
        },
        YOLOVersion::V10 => match args.task {
            YOLOTask::Detect => options.with_model("yolov10n-dyn.onnx")?,
            t => todo!("{t:?} is unsupported for {:?}", args.version),
        },
    }
    .with_yolo_version(args.version)
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
        .with_i00((1, 1, 4).into())
        .with_i02((args.height_min, args.height, args.height_max).into())
        .with_i03((args.width_min, args.width, args.width_max).into())
        .with_confs(&[0.4, 0.15]) // class_0: 0.4, others: 0.15
        .with_names2(&coco::KEYPOINTS_NAMES_17)
        .with_profile(args.profile);
    let mut model = YOLO::new(options)?;

    // build dataloader
    let dl = DataLoader::default()
        .with_batch(model.batch() as _)
        .load(args.source)?;

    // build annotator
    let annotator = Annotator::default()
        .with_skeletons(&coco::SKELETONS_16)
        .with_bboxes_thickness(7)
        .without_masks(true) // No masks plotting.
        .with_saveout("YOLO-Series");

    // run & annotate
    for (xs, _paths) in dl {
        // let ys = model.run(&xs)?;  // way one
        let ys = model.forward(&xs, args.profile)?; // way two

        if args.plot {
            annotator.annotate(&xs, &ys);
        } else {
            println!("{:?}", ys);
        }
    }

    Ok(())
}
