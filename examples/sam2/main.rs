use anyhow::Result;
use usls::{
    models::{SamPrompt, SAM2},
    Annotator, Config, DataLoader, Scale,
};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale
    #[argh(option, default = "String::from(\"t\")")]
    scale: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // Build model
    let config = match args.scale.parse()? {
        Scale::T => Config::sam2_1_tiny(),
        Scale::S => Config::sam2_1_small(),
        Scale::B => Config::sam2_1_base_plus(),
        Scale::L => Config::sam2_1_large(),
        _ => unimplemented!("Unsupported model scale: {:?}. Try b, s, t, l.", args.scale),
    }
    .with_device_all(args.device.parse()?)
    .commit()?;
    let mut model = SAM2::new(config)?;

    // Load image
    let xs = DataLoader::try_read_n(&["images/truck.jpg"])?;

    // Prompt
    let prompts = vec![SamPrompt::default()
        // //  # demo: point + point
        // .with_positive_point(500., 375.) // mid window
        // .with_positive_point(1125., 625.), // car door
        // // # demo: bbox
        // .with_xyxy(425., 600., 700., 875.), // left wheel
        // // # demo: bbox + negative point
        // .with_xyxy(425., 600., 700., 875.) // left wheel
        //     .with_negative_point(575., 750.), // tire
        // # demo: multiple objects with boxes
        .with_xyxy(75., 275., 1725., 850.)
        .with_xyxy(425., 600., 700., 875.)
        .with_xyxy(1375., 550., 1650., 800.)
        .with_xyxy(1240., 675., 1400., 750.)];

    // Run & Annotate
    let ys = model.forward(&xs, &prompts)?;

    // annotate
    let annotator = Annotator::default()
        .with_mask_style(usls::Style::mask().with_draw_mask_polygon_largest(true));

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
