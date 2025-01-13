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
    let options = Options::slanet_lcnet_v2_mobile_ch()
        .with_model_device(args.device.as_str().try_into()?)
        .with_model_dtype(args.dtype.as_str().try_into()?)
        .commit()?;
    let mut model = SLANet::new(options)?;

    // load
    let xs = DataLoader::try_read_batch(&[args.source])?;

    // run
    let ys = model.forward(&xs)?;
    // println!("{:?}", ys);

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
