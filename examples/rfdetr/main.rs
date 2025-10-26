use anyhow::Result;
use usls::{models::RFDETR, Annotator, Config, DataLoader, Scale, Task};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu\")")]
    device: String,

    /// task
    #[argh(option, default = "String::from(\"det\")")]
    task: String,

    /// scale
    #[argh(option, default = "String::from(\"n\")")]
    scale: String,

    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,
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
    .commit()?;
    let mut model = RFDETR::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;

    // extract bboxes
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default();
    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", model.spec()])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    // summary
    usls::perf(false);

    Ok(())
}
