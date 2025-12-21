use anyhow::Result;
use usls::{models::RFDETR, Annotator, Config, DataLoader, Scale, Task};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// source: image, image folder, video stream
    #[argh(option, default = "String::from(\"./assets/bus.jpg\")")]
    source: String,

    /// device
    #[argh(option, default = "String::from(\"cpu\")")]
    device: String,

    /// processor device
    #[argh(option, default = "String::from(\"cuda\")")]
    processor_device: String,

    /// task
    #[argh(option, default = "String::from(\"det\")")]
    task: String,

    /// scale
    #[argh(option, default = "String::from(\"n\")")]
    scale: String,

    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// batch size
    #[argh(option, default = "1")]
    batch_size: usize,

    /// confidences
    #[argh(option)]
    confs: Vec<f32>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // config
    let config = match args.task.parse()? {
        Task::ObjectDetection => match args.scale.parse()? {
            Scale::N => Config::rfdetr_nano(),
            Scale::S => Config::rfdetr_small(),
            Scale::M => Config::rfdetr_medium(),
            Scale::B => Config::rfdetr_base(),
            Scale::L => Config::rfdetr_large(),
            _ => unimplemented!("Unsupported model scale: {:?}. Try b, s, t.", args.scale),
        },
        Task::InstanceSegmentation => Config::rfdetr_seg_preview(),
        _ => unimplemented!("Unsupported task: {:?}. Try det, seg.", args.task),
    }
    .with_dtype_all(args.dtype.parse()?)
    .with_device_all(args.device.parse()?)
    .with_image_processor_device(args.processor_device.parse()?)
    .with_class_confs(if args.confs.is_empty() {
        &[0.35, 0.3]
    } else {
        &args.confs
    })
    .with_model_ixx(0, 0, (1, args.batch_size, args.batch_size))
    .commit()?;
    let mut model = RFDETR::new(config)?;

    // dataloader
    let dl = DataLoader::new(&args.source)?
        .with_batch(model.batch() as _)
        .build()?;

    // annotator
    let annotator = Annotator::default().with_mask_style(
        usls::MaskStyle::default()
            .with_visible(true)
            .with_cutout(true)
            .with_draw_polygon_largest(true),
    );

    // run & annotate
    for xs in &dl {
        let ys = model.forward(&xs)?;
        println!("ys: {:?}", ys);

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

    // summary
    usls::perf(false);

    Ok(())
}
