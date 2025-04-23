use anyhow::Result;
use usls::{models::Moondream2, Annotator, DataLoader, Options, Scale, Task};

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
    let (
        options_vision_encoder,
        options_vision_projection,
        options_text_decoder,
        options_text_encoder,
        options_coord_decoder,
        options_coord_encoder,
        options_size_decoder,
        options_size_encoder,
    ) = match args.scale.as_str().try_into()? {
        Scale::Billion(2.) => (
            Options::moondream2_2b_vision_encoder(),
            Options::moondream2_2b_vision_projection(),
            Options::moondream2_2b_text_decoder(),
            Options::moondream2_2b_text_encoder(),
            Options::moondream2_2b_coord_decoder(),
            Options::moondream2_2b_coord_encoder(),
            Options::moondream2_2b_size_decoder(),
            Options::moondream2_2b_size_encoder(),
        ),
        Scale::Billion(0.5) => (
            Options::moondream2_0_5b_vision_encoder(),
            Options::moondream2_0_5b_vision_projection(),
            Options::moondream2_0_5b_text_decoder(),
            Options::moondream2_0_5b_text_encoder(),
            Options::moondream2_0_5b_coord_decoder(),
            Options::moondream2_0_5b_coord_encoder(),
            Options::moondream2_0_5b_size_decoder(),
            Options::moondream2_0_5b_size_encoder(),
        ),
        _ => unimplemented!(),
    };

    let mut model = Moondream2::new(
        options_vision_encoder
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
        options_vision_projection
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
        options_text_encoder
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
        options_text_decoder
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
        Some(
            options_coord_encoder
                .with_model_dtype(args.dtype.as_str().try_into()?)
                .with_model_device(args.device.as_str().try_into()?)
                .commit()?,
        ),
        Some(
            options_coord_decoder
                .with_model_dtype(args.dtype.as_str().try_into()?)
                .with_model_device(args.device.as_str().try_into()?)
                .commit()?,
        ),
        Some(
            options_size_encoder
                .with_model_dtype(args.dtype.as_str().try_into()?)
                .with_model_device(args.device.as_str().try_into()?)
                .commit()?,
        ),
        Some(
            options_size_decoder
                .with_model_dtype(args.dtype.as_str().try_into()?)
                .with_model_device(args.device.as_str().try_into()?)
                .commit()?,
        ),
    )?;

    // load images
    let xs = DataLoader::try_read_n(&args.source)?;

    // run with task
    let task: Task = args.task.as_str().try_into()?;
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
            // let annotator = Annotator::default()
            //     .with_bboxes_thickness(4)
            //     .without_bboxes_conf(true)
            //     .with_keypoints_radius(6)
            //     .with_keypoints_name(true)
            //     .with_saveout("moondream2");
            // annotator.annotate(&xs, &ys);

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

    Ok(())
}
