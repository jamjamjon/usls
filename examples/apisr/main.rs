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

    /// kind: grl-4x, rrdb-2x
    #[argh(option, default = "String::from(\"rrdb-2x\")")]
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
        "rrdb-2x" => Config::apisr_rrdb_2x(),
        "grl-4x" => Config::apisr_grl_4x(),
        _ => unreachable!(),
    }
    .with_model_dtype(args.dtype.parse()?)
    .with_model_device(args.device.parse()?)
    .commit()?;
    let mut model = Swin2SR::new(config)?;

    // load image
    let xs = DataLoader::try_read_n(&["images/ekko.jpg"])?;

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
