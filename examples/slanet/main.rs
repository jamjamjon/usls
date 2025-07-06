use anyhow::Result;
use usls::{models::SLANet, Annotator, Color, Config, DataLoader};

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
    let config = Config::slanet_lcnet_v2_mobile_ch()
        .with_model_device(args.device.parse()?)
        .with_model_dtype(args.dtype.parse()?)
        .commit()?;
    let mut model = SLANet::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&[args.source])?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default().with_keypoint_style(
        usls::Style::keypoint()
            .with_text_visible(false)
            .with_skeleton(
                (
                    [(0, 1), (1, 2), (2, 3), (3, 0)],
                    [Color::black(), Color::red(), Color::green(), Color::blue()],
                )
                    .into(),
            ),
    );

    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", model.spec()])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    // summary
    usls::perf(false);

    Ok(())
}
