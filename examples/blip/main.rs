use usls::{models::Blip, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // visual
    let options_visual = Options::default()
        .with_model("blip/visual-base.onnx")?
        // .with_ixx(0, 2, 384.into())
        // .with_ixx(0, 3, 384.into())
        .with_profile(false);

    // textual
    let options_textual = Options::default()
        .with_model("blip/textual-base.onnx")?
        .with_tokenizer("blip/tokenizer.json")?
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
