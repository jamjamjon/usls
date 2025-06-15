use anyhow::Result;
use usls::{models::RTMO, Annotator, Config, DataLoader, Style, SKELETON_COCO_19};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// dtype
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale: t, s, m, l
    #[argh(option, default = "String::from(\"t\")")]
    scale: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = match args.scale.as_str() {
        "t" => Config::rtmo_t(),
        "s" => Config::rtmo_s(),
        "m" => Config::rtmo_m(),
        "l" => Config::rtmo_l(),
        _ => unreachable!(),
    }
    .with_model_dtype(args.dtype.parse()?)
    .with_model_device(args.device.parse()?)
    .commit()?;
    let mut model = RTMO::new(config)?;

    // load image
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;
    // println!("ys: {:?}", ys);

    // annotate
    let annotator = Annotator::default()
        .with_hbb_style(Style::hbb().with_draw_fill(true))
        .with_keypoint_style(
            Style::keypoint()
                .with_skeleton(SKELETON_COCO_19.into())
                .show_confidence(false)
                .show_id(true)
                .show_name(true),
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
