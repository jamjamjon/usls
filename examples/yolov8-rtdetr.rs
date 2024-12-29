use anyhow::Result;
use usls::{models::YOLO, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args: Args = argh::from_env();

    // build model
    let config = Options::yolo_v8_rtdetr_l()
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = YOLO::new(config)?;

    // load images
    let xs = DataLoader::try_read_batch(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    model.summary();

    Ok(())
}
