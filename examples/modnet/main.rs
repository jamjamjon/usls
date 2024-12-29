use usls::{models::MODNet, Annotator, DataLoader, Options};

fn main() -> anyhow::Result<()> {
    // build model
    let options = Options::modnet_photographic().commit()?;
    let mut model = MODNet::new(options)?;

    // load image
    let xs = [DataLoader::try_read("images/liuyifei.png")?];

    // run
    let ys = model.forward(&xs)?;

    // annotate
    let annotator = Annotator::default().with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    Ok(())
}
