use anyhow::Result;
use usls::{models::Swin2SR, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// kind: lightweight, classical-2x, classical-4x, compressed, realworld
    #[argh(option, default = "String::from(\"lightweight\")")]
    kind: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = match args.kind.as_str() {
        "lightweight" => Config::swin2sr_lightweight_x2_64(),
        "classical-2x" => Config::swin2sr_classical_x2_64(),
        "classical-4x" => Config::swin2sr_classical_x4_64(),
        "compressed" => Config::swin2sr_compressed_x4_48(),
        "realworld" => Config::swin2sr_realworld_x4_64_bsrgan_psnr(),
        _ => unreachable!(),
    }
    .with_model_dtype(args.dtype.parse()?)
    .with_model_device(args.device.parse()?)
    .commit()?;
    let mut model = Swin2SR::new(config)?;

    // load image
    let xs = DataLoader::try_read_n(&["images/butterfly.jpg"])?;

    // run
    let ys = model.forward(&xs)?;

    // save
    for y in ys {
        if let Some(images) = y.images() {
            for image in images.iter() {
                image.save(format!(
                    "{}.png",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs", model.spec()])?
                        .join(usls::timestamp(None))
                        .display(),
                ))?;
            }
        }
    }
    usls::perf(false);

    Ok(())
}
