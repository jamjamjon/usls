use anyhow::Result;
use usls::{models::Florence2, Annotator, DataLoader, Options, Scale, Task};

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
    let args: Args = argh::from_env();

    // load images
    let xs = [
        DataLoader::try_read("images/green-car.jpg")?,
        DataLoader::try_read("assets/bus.jpg")?,
    ];

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
        Task::OpenSetDetection("a vehicle"),
        Task::CaptionToPhraseGrounding("A vehicle with two wheels parked in front of a building."),
        Task::ReferringExpressionSegmentation("a vehicle"),
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

    // annotator
    let annotator = Annotator::new()
        .without_bboxes_conf(true)
        .with_bboxes_thickness(3)
        .with_saveout_subs(&["Florence2"]);

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
                let annotator = annotator.clone().with_saveout("Dense-Region-Caption");
                annotator.annotate(&xs, &ys);
            }
            Task::RegionProposal => {
                let annotator = annotator
                    .clone()
                    .without_bboxes_name(false)
                    .with_saveout("Region-Proposal");

                annotator.annotate(&xs, &ys);
            }
            Task::ObjectDetection => {
                let annotator = annotator.clone().with_saveout("Object-Detection");
                annotator.annotate(&xs, &ys);
            }
            Task::OpenSetDetection(_) => {
                let annotator = annotator.clone().with_saveout("Open-Set-Detection");
                annotator.annotate(&xs, &ys);
            }
            Task::CaptionToPhraseGrounding(_) => {
                let annotator = annotator
                    .clone()
                    .with_saveout("Caption-To-Phrase-Grounding");
                annotator.annotate(&xs, &ys);
            }
            Task::ReferringExpressionSegmentation(_) => {
                let annotator = annotator
                    .clone()
                    .with_saveout("Referring-Expression-Segmentation");
                annotator.annotate(&xs, &ys);
            }
            Task::RegionToSegmentation(..) => {
                let annotator = annotator.clone().with_saveout("Region-To-Segmentation");
                annotator.annotate(&xs, &ys);
            }
            Task::OcrWithRegion => {
                let annotator = annotator.clone().with_saveout("Ocr-With-Region");
                annotator.annotate(&xs, &ys);
            }

            _ => (),
        }
    }

    model.summary();

    Ok(())
}
