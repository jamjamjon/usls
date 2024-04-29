use usls::{models::Blip, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // visual
    let options_visual = Options::default()
        .with_model("../models/blip-visual-base.onnx")?
        .with_i00((1, 1, 4).into())
        .with_profile(false);

    // textual
    let options_textual = Options::default()
        .with_model("../models/blip-textual-base.onnx")?
        .with_tokenizer("tokenizer-blip.json")?
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
    let x = vec![DataLoader::try_read("./assets/bus.jpg")?];
    let _y = model.caption(&x, None, true)?; // unconditional
    let y = model.caption(&x, Some("three man"), true)?; // conditional
    println!("{:?}", y[0].texts());

    Ok(())
}
