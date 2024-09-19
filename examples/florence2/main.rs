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
    // .with_task(Task::Caption(2));

    // load images
    let xs = [
        DataLoader::try_read("florence2/car.jpg")?,
        DataLoader::try_read("assets/bus.jpg")?,
    ];

    // run with a batch of tasks
    let ys = model.run(
        &xs,
        &[
            // w/ inputs
            Task::Caption(0),
            Task::Caption(1),
            Task::Caption(2),
            Task::Ocr,
            Task::RegionProposal,
            Task::ObjectDetection,
            Task::DenseRegionCaption,
            // // Task::OcrWithRegion, // TODO
            // w/o inputs
            // Task::OpenSetDetection("A green car".into()),
            // Task::CaptionToPhraseGrounding("A green car".into()),
            // Task::ReferringExpressionSegmentation("A green car".into()),
            // Task::RegionToSegmentation(702, 575, 866, 772),
            // Task::RegionToCategory(52, 332, 932, 774),
            // Task::RegionToDescription(52, 332, 932, 774),
            // Task::RegionToOcr(100, 100, 300, 300),
        ],
    )?;

    // annotator
    for (task, ys_) in ys.iter() {
        match task {
            Task::Caption(_) | Task::Ocr => println!("Task: {:?}\n{:?}\n", task, ys_),
            Task::DenseRegionCaption => {
                let annotator = Annotator::default()
                    .without_bboxes_conf(true)
                    .with_bboxes_thickness(4)
                    .with_saveout("Florence2-DenseRegionCaption");
                annotator.annotate(&xs, ys_);
            }
            Task::RegionProposal => {
                let annotator = Annotator::default()
                    .without_bboxes_conf(true)
                    .without_bboxes_name(true)
                    .with_bboxes_thickness(4)
                    .with_saveout("Florence2-RegionProposal");
                annotator.annotate(&xs, ys_);
            }
            Task::ObjectDetection => {
                let annotator = Annotator::default()
                    .without_bboxes_conf(true)
                    .with_bboxes_thickness(4)
                    .with_saveout("Florence2-ObjectDetection");
                annotator.annotate(&xs, ys_);
            }

            _ => (),
        }
    }

    Ok(())
}
