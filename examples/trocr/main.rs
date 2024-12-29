use usls::{
    models::{TrOCR, TrOCRKind},
    DataLoader, Options, Scale,
};

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
    #[argh(option, default = "String::from(\"s\")")]
    scale: String,

    /// kind
    #[argh(option, default = "String::from(\"printed\")")]
    kind: String,
}

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();

    // load images
    let xs = DataLoader::try_read_batch(&[
        "images/text-en-dark.png",
        "images/text-hello-rust-handwritten.png",
    ])?;

    // build model
    let (options_encoder, options_decoder, options_decoder_merged) =
        match args.scale.as_str().try_into()? {
            Scale::S => match args.kind.as_str().try_into()? {
                TrOCRKind::Printed => (
                    Options::trocr_encoder_small_printed(),
                    Options::trocr_decoder_small_printed(),
                    Options::trocr_decoder_merged_small_printed(),
                ),
                TrOCRKind::HandWritten => (
                    Options::trocr_encoder_small_handwritten(),
                    Options::trocr_decoder_small_handwritten(),
                    Options::trocr_decoder_merged_small_handwritten(),
                ),
            },
            Scale::B => match args.kind.as_str().try_into()? {
                TrOCRKind::Printed => (
                    Options::trocr_encoder_base_printed(),
                    Options::trocr_decoder_base_printed(),
                    Options::trocr_decoder_merged_base_printed(),
                ),
                TrOCRKind::HandWritten => (
                    Options::trocr_encoder_base_handwritten(),
                    Options::trocr_decoder_base_handwritten(),
                    Options::trocr_decoder_merged_base_handwritten(),
                ),
            },
            x => anyhow::bail!("Unsupported TrOCR scale: {:?}", x),
        };

    let mut model = TrOCR::new(
        options_encoder
            .with_model_device(args.device.as_str().try_into()?)
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
        options_decoder
            .with_model_device(args.device.as_str().try_into()?)
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
        options_decoder_merged
            .with_model_device(args.device.as_str().try_into()?)
            .with_model_dtype(args.dtype.as_str().try_into()?)
            .with_batch_size(xs.len())
            .commit()?,
    )?;

    // inference
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // summary
    model.summary();

    Ok(())
}
