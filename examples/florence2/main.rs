use anyhow::Result;
use usls::{models::Florence2, Annotator, DataLoader, Options, Scale, Style, Task};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale
    #[argh(option, default = "String::from(\"base\")")]
    scale: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // load images
    let xs = DataLoader::try_read_n(&["images/green-car.jpg", "assets/bus.jpg"])?;

    // build model
    let (
        options_vision_encoder,
        options_text_embed,
        options_encoder,
        options_decoder,
        options_decoder_merged,
    ) = match args.scale.as_str().try_into()? {
        Scale::B => (
            Options::florence2_visual_encoder_base(),
            Options::florence2_textual_embed_base(),
            Options::florence2_texual_encoder_base(),
            Options::florence2_texual_decoder_base(),
            Options::florence2_texual_decoder_merged_base(),
        ),
        Scale::L => todo!(),
        _ => anyhow::bail!("Unsupported Florence2 scale."),
    };

    let mut model = Florence2::new(
        options_vision_encoder
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
        options_text_embed
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
        options_encoder
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
        options_decoder
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
        options_decoder_merged
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_model_device(args.device.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
    )?;

    // tasks
    let tasks = [
        // w inputs
        Task::Caption(0),
        Task::Caption(1),
        Task::Caption(2),
        Task::Ocr,
        // Task::OcrWithRegion,
        Task::RegionProposal,
        Task::ObjectDetection,
        Task::DenseRegionCaption,
        // w/o inputs
        Task::OpenSetDetection("a vehicle".into()),
        Task::CaptionToPhraseGrounding(
            "A vehicle with two wheels parked in front of a building.".into(),
        ),
        Task::ReferringExpressionSegmentation("a vehicle".into()),
        Task::RegionToSegmentation(
            // 31, 156, 581, 373,  // car
            449, 270, 556, 372, // wheel
        ),
        Task::RegionToCategory(
            // 31, 156, 581, 373,
            449, 270, 556, 372,
        ),
        Task::RegionToDescription(
            // 31, 156, 581, 373,
            449, 270, 556, 372,
        ),
    ];

    // inference
    for task in tasks.iter() {
        let ys = model.forward(&xs, task)?;

        // annotate
        match task {
            Task::Caption(_)
            | Task::Ocr
            | Task::RegionToCategory(..)
            | Task::RegionToDescription(..) => {
                println!("Task: {:?}\n{:?}\n", task, &ys)
            }
            Task::DenseRegionCaption => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default()
                        .with_hbb_style(Style::hbb().show_confidence(false))
                        .annotate(x, y)?
                        .save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&["runs", "Florence2", "Dense-Region-Caption"])?
                                .join(usls::timestamp(None))
                                .display(),
                        ))?;
                }
            }
            Task::RegionProposal => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default()
                        .with_hbb_style(Style::hbb().show_confidence(false).show_name(false))
                        .annotate(x, y)?
                        .save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&["runs", "Florence2", "Region-Proposal"])?
                                .join(usls::timestamp(None))
                                .display(),
                        ))?;
                }
            }
            Task::ObjectDetection => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default().annotate(x, y)?.save(format!(
                        "{}.jpg",
                        usls::Dir::Current
                            .base_dir_with_subs(&["runs", "Florence2", "Object-Detection"])?
                            .join(usls::timestamp(None))
                            .display(),
                    ))?;
                }
            }
            Task::OpenSetDetection(_) => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default().annotate(x, y)?.save(format!(
                        "{}.jpg",
                        usls::Dir::Current
                            .base_dir_with_subs(&["runs", "Florence2", "Open-Object-Detection"])?
                            .join(usls::timestamp(None))
                            .display(),
                    ))?;
                }
            }
            Task::CaptionToPhraseGrounding(_) => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default()
                        .with_hbb_style(Style::hbb().show_confidence(false))
                        .annotate(x, y)?
                        .save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&[
                                    "runs",
                                    "Florence2",
                                    "Caption-To-Phrase-Grounding"
                                ])?
                                .join(usls::timestamp(None))
                                .display(),
                        ))?;
                }
            }
            Task::ReferringExpressionSegmentation(_) => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default()
                        .with_hbb_style(Style::hbb().show_confidence(false))
                        .annotate(x, y)?
                        .save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&[
                                    "runs",
                                    "Florence2",
                                    "Referring-Expression-Segmentation"
                                ])?
                                .join(usls::timestamp(None))
                                .display(),
                        ))?;
                }
            }
            Task::RegionToSegmentation(..) => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default()
                        .with_hbb_style(Style::hbb().show_confidence(false))
                        .annotate(x, y)?
                        .save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&[
                                    "runs",
                                    "Florence2",
                                    "Region-To-Segmentation",
                                ])?
                                .join(usls::timestamp(None))
                                .display(),
                        ))?;
                }
            }
            Task::OcrWithRegion => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    Annotator::default()
                        .with_hbb_style(Style::hbb().show_confidence(false))
                        .annotate(x, y)?
                        .save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&["runs", "Florence2", "Ocr-With-Region",])?
                                .join(usls::timestamp(None))
                                .display(),
                        ))?;
                }
            }

            _ => (),
        }
    }

    model.summary();

    Ok(())
}
