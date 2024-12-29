use anyhow::Result;
use usls::{models::SLANet, Annotator, DataLoader, Options};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// source
    #[argh(option, default = "String::from(\"images/table.png\")")]
    source: String,

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
    let options = Options::slanet_lcnet_v2_mobile_ch()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = SLANet::new(options)?;

    // load
    let xs = DataLoader::try_read_batch(&[args.source])?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default()
        .with_keypoints_radius(2)
        .with_skeletons(&[(0, 1), (1, 2), (2, 3), (3, 0)])
        .with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    // summary
    model.summary();

    Ok(())
}
