use anyhow::Result;
use usls::{models::SVTR, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build model
    let options = Options::ppocr_rec_v4_ch()
        // ppocr_rec_v4_en()
        // repsvtr_ch()
        .with_model_device(args.device.as_str().try_into()?)
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .commit()?;
    let mut model = SVTR::new(options)?;

    // load images
    let dl = DataLoader::new("./examples/svtr/images")?
        .with_batch(model.batch() as _)
        .with_progress_bar(false)
        .build()?;

    // run
    for xs in &dl {
        let ys = model.forward(&xs)?;
        println!("ys: {:?}", ys);
    }

    // summary
    model.summary();

    Ok(())
}
