use usls::{models::Blip, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // visual
    let options_visual = Options::default()
        .with_model("blip/visual-base.onnx")?
        .with_i00((1, 1, 4).into())
        .with_profile(false);

    // textual
    let options_textual = Options::default()
        .with_model("blip/textual-base.onnx")?
        // .with_tokenizer("blip/tokenizer.json")?
        .with_i00((1, 1, 4).into()) // input_id: batch
        .with_i01((1, 1, 4).into()) // input_id: seq_len
        .with_i10((1, 1, 4).into()) // attention_mask: batch
        .with_i11((1, 1, 4).into()) // attention_mask: seq_len
        .with_i20((1, 1, 4).into()) // encoder_hidden_states: batch
        .with_i30((1, 1, 4).into()) // encoder_attention_mask: batch
        .with_profile(false);

    // build model
    let mut model = Blip::new(options_visual, options_textual)?;

    // image caption (this demo use batch_size=1)
    let xs = [DataLoader::try_read("images/bus.jpg")?];
    let image_embeddings = model.encode_images(&xs)?;
    let _y = model.caption(&image_embeddings, None, true)?; // unconditional
    let y = model.caption(&image_embeddings, Some("three man"), true)?; // conditional
    println!("{:?}", y[0].texts());

    Ok(())
}
