use anyhow::Result;
use usls::DataLoader;
use usls::{models::PicoDet, Annotator, Config};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu\")")]
    device: String,

    /// processor device
    #[argh(option, default = "String::from(\"cpu\")")]
    processor_device: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // config
    let config = Config::picodet_layout_1x()
        // picodet_l_layout_3cls()
        // picodet_l_layout_17cls()
        .with_device_all(args.device.parse()?)
        .with_image_processor_device(args.processor_device.parse()?)
        .commit()?;
    let mut model = PicoDet::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&["images/academic.jpg"])?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default();
    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", model.spec()])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }
    usls::perf(false);

    Ok(())
}
