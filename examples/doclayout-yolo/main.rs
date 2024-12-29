use anyhow::Result;
use usls::{models::YOLO, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
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
    let config = Options::doclayout_yolo_docstructbench()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = YOLO::new(config)?;

    // load images
    let xs = [DataLoader::try_read("images/academic.jpg")?];

    // run
    let ys = model.forward(&xs)?;
    // println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout("doclayout-yolo");
    annotator.annotate(&xs, &ys);

    model.summary();

    Ok(())
}
