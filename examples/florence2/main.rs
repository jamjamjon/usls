use usls::{models::Florence2, DataLoader, Options, Task};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // vision encoder
    let options_vision_encoder = Options::default()
        .with_model("florence2/base-ft-vision-encoder-dyn.onnx")?
        .with_i00((1, 1, 4).into())
        .with_i02((512, 768, 800).into())
        .with_i03((512, 768, 800).into())
        .with_profile(false)
        .with_cuda(0);

    // text embed
    let options_text_embed = Options::default()
        .with_model("florence2/base-ft-embed-tokens-dyn.onnx")?
        .with_i00((1, 1, 4).into())
        .with_i01((1, 1, 20).into()) // seq_length
        .with_tokenizer("florence2/tokenizer.json")?
        .with_profile(false);

    // transformer encoder
    let options_encoder = Options::default()
        .with_model("florence2/base-ft-encoder.onnx")?
        .with_i00((1, 1, 4).into())
        .with_i01((1, 1, 300).into()) // encoder_sequence_length
        .with_i10((1, 1, 4).into())
        .with_i11((1, 1, 300).into()) // encoder_sequence_length
        .with_profile(false);

    // transformer decoder
    let options_decoder = Options::default()
        .with_model("florence2/base-ft-decoder-dyn.onnx")?
        .with_i00((1, 1, 4).into())
        .with_i01((1, 1, 300).into()) // encoder_sequence_length
        .with_i10((1, 1, 4).into())
        .with_i11((1, 1, 300).into()) // encoder_sequence_length
        .with_i20((1, 1, 4).into())
        .with_i21((1, 1, 300).into()) // encoder_sequence_length
        .with_profile(false);

    // transformer decoder merged
    let options_decoder_merged = Options::default()
        .with_model("florence2/base-ft-decoder-merged-dyn.onnx")?
        // encoder_attention_mask
        .with_i00((1, 1, 4).into())
        .with_i01((1, 1, 300).into()) // encoder_sequence_length
        // encoder_hidden_states
        .with_i10((1, 1, 4).into())
        .with_i11((1, 1, 300).into()) // encoder_sequence_length
        // inputs_embeds
        .with_i20((1, 1, 4).into())
        .with_i21((1, 1, 300).into()) // encoder_sequence_length
        // past_key_values.0.decoder.key
        .with_i30((1, 1, 4).into())
        .with_i32_((1, 1, 1).into())
        // past_key_values.0.decoder.value
        .with_i40((1, 1, 4).into())
        .with_i42((1, 1, 1).into())
        // past_key_values.0.encoder.key
        .with_i50((1, 1, 4).into())
        .with_i52((1, 1, 1).into())
        // past_key_values.0.decoder.value
        .with_i60((1, 1, 4).into())
        .with_i62((1, 1, 1).into())
        // past_key_values.1.decoder.key
        .with_i70((1, 1, 4).into())
        .with_i72((1, 1, 1).into())
        // past_key_values.1.decoder.value
        .with_i80((1, 1, 4).into())
        .with_i82((1, 1, 1).into())
        // past_key_values.1.encoder.key
        .with_i90((1, 1, 4).into())
        .with_i92((1, 1, 1).into())
        // past_key_values.1.decoder.value
        .with_i100((1, 1, 4).into())
        .with_i102((1, 1, 1).into())
        // past_key_values.2.decoder.key
        .with_i110((1, 1, 4).into())
        .with_i112((1, 1, 1).into())
        // past_key_values.2.decoder.value
        .with_i120((1, 1, 4).into())
        .with_i122((1, 1, 1).into())
        // past_key_values.2.encoder.key
        .with_i130((1, 1, 4).into())
        .with_i132((1, 1, 1).into())
        // past_key_values.2.decoder.value
        .with_i140((1, 1, 4).into())
        .with_i142((1, 1, 1).into())
        // past_key_values.3.decoder.key
        .with_i150((1, 1, 4).into())
        .with_i152((1, 1, 1).into())
        // past_key_values.3.decoder.value
        .with_i160((1, 1, 4).into())
        .with_i162((1, 1, 1).into())
        // past_key_values.3.encoder.key
        .with_i170((1, 1, 4).into())
        .with_i172((1, 1, 1).into())
        // past_key_values.3.decoder.value
        .with_i180((1, 1, 4).into())
        .with_i182((1, 1, 1).into())
        // past_key_values.4.decoder.key
        .with_i190((1, 1, 4).into())
        .with_i192((1, 1, 1).into())
        // past_key_values.4.decoder.value
        .with_i200((1, 1, 4).into())
        .with_i202((1, 1, 1).into())
        // past_key_values.4.encoder.key
        .with_i210((1, 1, 4).into())
        .with_i212((1, 1, 1).into())
        // past_key_values.4.decoder.value
        .with_i220((1, 1, 4).into())
        .with_i222((1, 1, 1).into())
        // past_key_values.5.decoder.key
        .with_i230((1, 1, 4).into())
        .with_i232((1, 1, 1).into())
        // past_key_values.5.decoder.value
        .with_i240((1, 1, 4).into())
        .with_i242((1, 1, 1).into())
        // past_key_values.5.encoder.key
        .with_i250((1, 1, 4).into())
        .with_i252((1, 1, 1).into())
        // past_key_values.5.decoder.value
        .with_i260((1, 1, 4).into())
        .with_i262((1, 1, 1).into())
        //use_cache_branch
        .with_i270((1, 1, 1).into())
        .with_profile(false);

    // build model
    let mut model = Florence2::new(
        options_vision_encoder,
        options_text_embed,
        options_encoder,
        options_decoder,
        options_decoder_merged,
    )?
    .with_task(Task::Caption(2));

    // load images
    let images = [DataLoader::try_read("florence2/car.jpg")?];

    // encode image
    let image_embeddings = model.encode_images(&images)?;

    // caption
    let _ys = model.caption(&image_embeddings, true)?; // display results

    Ok(())
}
