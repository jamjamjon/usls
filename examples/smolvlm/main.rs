use anyhow::Result;
use usls::{models::SmolVLM, DataLoader, Options, Scale};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image
    #[argh(option, default = "vec![String::from(\"./assets/bus.jpg\")]")]
    source: Vec<String>,

    /// promt
    #[argh(option, default = "String::from(\"Can you describe this image?\")")]
    prompt: String,

    /// scale
    #[argh(option, default = "String::from(\"256m\")")]
    scale: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let (options_vision_encoder, options_text_embed, options_decode) =
        match args.scale.as_str().try_into()? {
            Scale::Million(256.) => (
                Options::smolvlm_vision_256m(),
                Options::smolvlm_text_embed_256m(),
                Options::smolvlm_decoder_256m(),
            ),
            Scale::Million(500.) => (
                Options::smolvlm_vision_500m(),
                Options::smolvlm_text_embed_500m(),
                Options::smolvlm_decoder_500m(),
            ),
            _ => unimplemented!(),
        };

    let mut model = SmolVLM::new(
        options_vision_encoder
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
        options_text_embed
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
        options_decode
            .with_model_device(args.device.as_str().try_into()?)
            .commit()?,
    )?;

    // load images
    let xs = DataLoader::try_read_batch(&args.source)?;

    // run
    let ys = model.forward(&xs, &args.prompt)?;

    for y in ys.iter() {
        if let Some(texts) = y.texts() {
            for text in texts {
                println!("[User]: {}\n\n[Assistant]:{}", args.prompt, text);
            }
        }
    }

    Ok(())
}
