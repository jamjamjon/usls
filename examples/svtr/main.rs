use anyhow::Result;
use usls::{models::SVTR, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // build model
    let options = Options::ppocr_rec_v4_ch()
        // svtr_v2_teacher_ch()
        // .with_batch_size(2)
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = SVTR::new(options)?;

    // load images
    let dl = DataLoader::new("./examples/svtr/images")?
        .with_batch(model.batch() as _)
        .with_progress_bar(false)
        .build()?;

    // run
    for (xs, paths) in dl {
        let ys = model.forward(&xs)?;
        println!("{paths:?}: {:?}", ys)
    }

    //summary
    model.summary();

    Ok(())
}
