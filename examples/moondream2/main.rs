use anyhow::Result;
use usls::{models::Moondream2, Annotator, Config, DataLoader, Scale, Task};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image
    #[argh(
        option,
        default = "vec![
            String::from(\"./assets/bus.jpg\"),
            String::from(\"images/green-car.jpg\"),
        ]"
    )]
    source: Vec<String>,

    /// dtype
    #[argh(option, default = "String::from(\"int4\")")]
    dtype: String,

    /// scale
    #[argh(option, default = "String::from(\"0.5b\")")]
    scale: String,

    /// task
    #[argh(option, default = "String::from(\"Caption: 0\")")]
    task: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = match args.scale.parse()? {
        Scale::Billion(0.5) => Config::moondream2_0_5b(),
        Scale::Billion(2.) => Config::moondream2_2b(),
        _ => unimplemented!(),
    }
    .with_dtype_all(args.dtype.parse()?)
    .with_device_all(args.device.parse()?)
    .commit()?;

    let mut model = Moondream2::new(config)?;

    // load images
    let xs = DataLoader::try_read_n(&args.source)?;

    // run with task
    let task: Task = args.task.parse()?;
    let ys = model.forward(&xs, &task)?;

    // annotate
    match task {
        Task::Caption(_) => {
            println!("{}:", task);
            for (i, y) in ys.iter().enumerate() {
                if let Some(texts) = y.texts() {
                    println!("Image {}: {:?}\n", i, texts[0]);
                }
            }
        }
        Task::Vqa(query) => {
            println!("Question: {}", query);
            for (i, y) in ys.iter().enumerate() {
                if let Some(texts) = y.texts() {
                    println!("Image {}: {:?}\n", i, texts[0]);
                }
            }
        }
        Task::OpenSetDetection(_) | Task::OpenSetKeypointsDetection(_) => {
            println!("{:?}", ys);

            // annotate
            let annotator = Annotator::default()
                .with_hbb_style(
                    usls::Style::hbb()
                        .with_draw_fill(true)
                        .show_confidence(false),
                )
                .with_keypoint_style(
                    usls::Style::keypoint()
                        .show_confidence(false)
                        .show_id(true)
                        .show_name(false),
                );

            for (x, y) in xs.iter().zip(ys.iter()) {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs", "moondream2"])?
                        .join(usls::timestamp(None))
                        .display(),
                ))?;
            }
        }
        _ => unimplemented!("Unsupported moondream2 task."),
    }
    usls::perf(false);

    Ok(())
}
