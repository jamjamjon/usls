use usls::{models::Florence2, Annotator, DataLoader, Options, Task};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // vision encoder
    let options_vision_encoder = Options::default()
        .with_model("florence2/base-vision-encoder.onnx")?
        .with_i00((1, 2, 4).into())
        .with_i02((512, 768, 800).into())
        .with_i03((512, 768, 800).into())
        .with_profile(false)
        .with_cuda(0);

    // text embed
    let options_text_embed = Options::default()
        .with_model("florence2/base-embed-tokens.onnx")?
        .with_i00((1, 2, 4).into())
        .with_i01((1, 2, 20).into()) // seq_length
        .with_tokenizer("florence2/tokenizer.json")?
        .with_profile(false);

    // transformer encoder
    let options_encoder = Options::default()
        .with_model("florence2/base-encoder.onnx")?
        .with_i00((1, 2, 4).into())
        .with_i01((1, 2, 300).into()) // encoder_sequence_length
        .with_i10((1, 2, 4).into())
        .with_i11((1, 2, 300).into()) // encoder_sequence_length
        .with_profile(false);

    // transformer decoder
    let options_decoder = Options::default()
        .with_model("florence2/base-decoder.onnx")?
        .with_i00((1, 2, 4).into())
        .with_i01((1, 2, 300).into()) // encoder_sequence_length
        .with_i10((1, 2, 4).into())
        .with_i11((1, 2, 300).into()) // encoder_sequence_length
        .with_i20((1, 2, 4).into())
        .with_i21((1, 2, 300).into()) // encoder_sequence_length
        .with_profile(false);

    // transformer decoder merged
    let options_decoder_merged = Options::default()
        .with_model("florence2/base-decoder-merged.onnx")?
        // encoder_attention_mask
        .with_i00((1, 2, 4).into())
        .with_i01((1, 2, 300).into()) // encoder_sequence_length
        // encoder_hidden_states
        .with_i10((1, 2, 4).into())
        .with_i11((1, 2, 300).into()) // encoder_sequence_length
        // inputs_embeds
        .with_i20((1, 2, 4).into())
        .with_i21((1, 2, 300).into()) // encoder_sequence_length
        // past_key_values.0.decoder.key
        .with_i30((1, 2, 4).into())
        .with_i32_((1, 2, 1).into())
        // past_key_values.0.decoder.value
        .with_i40((1, 2, 4).into())
        .with_i42((1, 2, 1).into())
        // past_key_values.0.encoder.key
        .with_i50((1, 2, 4).into())
        .with_i52((1, 2, 1).into())
        // past_key_values.0.decoder.value
        .with_i60((1, 2, 4).into())
        .with_i62((1, 2, 1).into())
        // past_key_values.1.decoder.key
        .with_i70((1, 2, 4).into())
        .with_i72((1, 2, 1).into())
        // past_key_values.1.decoder.value
        .with_i80((1, 2, 4).into())
        .with_i82((1, 2, 1).into())
        // past_key_values.1.encoder.key
        .with_i90((1, 2, 4).into())
        .with_i92((1, 2, 1).into())
        // past_key_values.1.decoder.value
        .with_i100((1, 2, 4).into())
        .with_i102((1, 2, 1).into())
        // past_key_values.2.decoder.key
        .with_i110((1, 2, 4).into())
        .with_i112((1, 2, 1).into())
        // past_key_values.2.decoder.value
        .with_i120((1, 2, 4).into())
        .with_i122((1, 2, 1).into())
        // past_key_values.2.encoder.key
        .with_i130((1, 2, 4).into())
        .with_i132((1, 2, 1).into())
        // past_key_values.2.decoder.value
        .with_i140((1, 2, 4).into())
        .with_i142((1, 2, 1).into())
        // past_key_values.3.decoder.key
        .with_i150((1, 2, 4).into())
        .with_i152((1, 2, 1).into())
        // past_key_values.3.decoder.value
        .with_i160((1, 2, 4).into())
        .with_i162((1, 2, 1).into())
        // past_key_values.3.encoder.key
        .with_i170((1, 2, 4).into())
        .with_i172((1, 2, 1).into())
        // past_key_values.3.decoder.value
        .with_i180((1, 2, 4).into())
        .with_i182((1, 2, 1).into())
        // past_key_values.4.decoder.key
        .with_i190((1, 2, 4).into())
        .with_i192((1, 2, 1).into())
        // past_key_values.4.decoder.value
        .with_i200((1, 2, 4).into())
        .with_i202((1, 2, 1).into())
        // past_key_values.4.encoder.key
        .with_i210((1, 2, 4).into())
        .with_i212((1, 2, 1).into())
        // past_key_values.4.decoder.value
        .with_i220((1, 2, 4).into())
        .with_i222((1, 2, 1).into())
        // past_key_values.5.decoder.key
        .with_i230((1, 2, 4).into())
        .with_i232((1, 2, 1).into())
        // past_key_values.5.decoder.value
        .with_i240((1, 2, 4).into())
        .with_i242((1, 2, 1).into())
        // past_key_values.5.encoder.key
        .with_i250((1, 2, 4).into())
        .with_i252((1, 2, 1).into())
        // past_key_values.5.decoder.value
        .with_i260((1, 2, 4).into())
        .with_i262((1, 2, 1).into())
        //use_cache_branch
        .with_i270((1, 2, 1).into())
        .with_profile(false);

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
                    .with_polygons_alpha(200)
                    .with_saveout("Referring-Expression-Segmentation");
                annotator.annotate(&xs, ys_);
            }
            Task::RegionToSegmentation(..) => {
                let annotator = annotator
                    .clone()
                    .with_polygons_alpha(200)
                    .with_saveout("Region-To-Segmentation");
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
