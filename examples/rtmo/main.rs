use anyhow::Result;
use usls::{models::RTMO, Annotator, DataLoader, Options, COCO_SKELETONS_16};

fn main() -> Result<()> {
    // build model
    let mut model = RTMO::new(Options::rtmo_s().commit()?)?;

    // load image
    let xs = [DataLoader::try_read("images/bus.jpg")?];

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default()
        .with_saveout(model.spec())
        .with_skeletons(&COCO_SKELETONS_16);
    annotator.annotate(&xs, &ys);

    Ok(())
}
