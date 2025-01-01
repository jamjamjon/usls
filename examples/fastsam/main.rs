use anyhow::Result;
use usls::{models::YOLO, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();

    // build model
    let config = Options::fastsam_s()
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = YOLO::new(config)?;

    // load images
    let xs = DataLoader::try_read_batch(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default()
        .without_masks(true)
        .with_bboxes_thickness(3)
        .with_saveout("fastsam");
    annotator.annotate(&xs, &ys);

    Ok(())
}
