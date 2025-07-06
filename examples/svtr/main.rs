use anyhow::Result;
use usls::{models::SVTR, Config, DataLoader};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// max text length
    #[argh(option, default = "960")]
    max_text_length: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build model
    let config = Config::ppocr_rec_v5_mobile()
        // ppocr_rec_v5_server()
        // ppocr_rec_v4_ch()
        // ppocr_rec_v4_en()
        // repsvtr_ch()
        .with_model_ixx(0, 3, args.max_text_length.into())
        .with_model_device(args.device.parse()?)
        .with_model_dtype(args.dtype.parse()?)
        .commit()?;
    let mut model = SVTR::new(config)?;

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
    usls::perf(false);

    Ok(())
}
