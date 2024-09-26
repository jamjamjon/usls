use usls::{models::Florence2, Annotator, DataLoader, Options, Task};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 3;

    // vision encoder
    let options_vision_encoder = Options::default()
        .with_model("florence2/base-vision-encoder-f16.onnx")?
        .with_ixx(0, 2, (512, 768, 800).into())
        .with_ixx(0, 3, 768.into())
        .with_ixx(0, 0, (1, batch_size as _, 8).into());

    // text embed
    let options_text_embed = Options::default()
        .with_model("florence2/base-embed-tokens-f16.onnx")?
        .with_tokenizer("florence2/tokenizer.json")?
        .with_batch(batch_size);

    // transformer encoder
    let options_encoder = Options::default()
        .with_model("florence2/base-encoder-f16.onnx")?
        .with_batch(batch_size);

    // transformer decoder
    let options_decoder = Options::default()
        .with_model("florence2/base-decoder-f16.onnx")?
        .with_batch(batch_size);

    // transformer decoder merged
    let options_decoder_merged = Options::default()
        .with_model("florence2/base-decoder-merged-f16.onnx")?
        .with_batch(batch_size);

    // build model
    let mut model = Florence2::new(
        options_vision_encoder,
        options_text_embed,
        options_encoder,
        options_decoder,
        options_decoder_merged,
    )?;

    // load images
    let xs = [
        // DataLoader::try_read("florence2/car.jpg")?, // for testing region-related tasks
        DataLoader::try_read("florence2/car.jpg")?,
        // DataLoader::try_read("images/db.png")?,
        DataLoader::try_read("assets/bus.jpg")?,
    ];

    // region-related tasks
    let quantizer = usls::Quantizer::default();
    // let coords = [449., 270., 556., 372.];  // wheel
    let coords = [31., 156., 581., 373.]; // car
    let (width_car, height_car) = (xs[0].width(), xs[0].height());
    let quantized_coords = quantizer.quantize(&coords, (width_car as _, height_car as _));

    // run with tasks
    let ys = model.run_with_tasks(
        &xs,
        &[
            // w/ inputs
            Task::Caption(0),
            Task::Caption(1),
            Task::Caption(2),
            Task::Ocr,
            Task::OcrWithRegion,
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
                quantized_coords[0],
                quantized_coords[1],
                quantized_coords[2],
                quantized_coords[3],
            ),
            Task::RegionToCategory(
                quantized_coords[0],
                quantized_coords[1],
                quantized_coords[2],
                quantized_coords[3],
            ),
            Task::RegionToDescription(
                quantized_coords[0],
                quantized_coords[1],
                quantized_coords[2],
                quantized_coords[3],
            ),
        ],
    )?;

    // annotator
    let annotator = Annotator::new()
        .without_bboxes_conf(true)
        .with_bboxes_thickness(3)
        .with_saveout_subs(&["Florence2"]);
    for (task, ys_) in ys.iter() {
        match task {
            Task::Caption(_)
            | Task::Ocr
            | Task::RegionToCategory(..)
            | Task::RegionToDescription(..) => {
                println!("Task: {:?}\n{:?}\n", task, ys_)
            }
            Task::DenseRegionCaption => {
                let annotator = annotator.clone().with_saveout("Dense-Region-Caption");
                annotator.annotate(&xs, ys_);
            }
            Task::RegionProposal => {
                let annotator = annotator
                    .clone()
                    .without_bboxes_name(false)
                    .with_saveout("Region-Proposal");

                annotator.annotate(&xs, ys_);
            }
            Task::ObjectDetection => {
                let annotator = annotator.clone().with_saveout("Object-Detection");
                annotator.annotate(&xs, ys_);
            }
            Task::OpenSetDetection(_) => {
                let annotator = annotator.clone().with_saveout("Open-Set-Detection");
                annotator.annotate(&xs, ys_);
            }
            Task::CaptionToPhraseGrounding(_) => {
                let annotator = annotator
                    .clone()
                    .with_saveout("Caption-To-Phrase-Grounding");
                annotator.annotate(&xs, ys_);
            }
            Task::ReferringExpressionSegmentation(_) => {
                let annotator = annotator
                    .clone()
                    .with_saveout("Referring-Expression-Segmentation");
                annotator.annotate(&xs, ys_);
            }
            Task::RegionToSegmentation(..) => {
                let annotator = annotator.clone().with_saveout("Region-To-Segmentation");
                annotator.annotate(&xs, ys_);
            }
            Task::OcrWithRegion => {
                let annotator = annotator.clone().with_saveout("Ocr-With-Region");
                annotator.annotate(&xs, ys_);
            }

            _ => (),
        }
    }

    Ok(())
}
