use anyhow::Result;
use usls::{models::vlm::YOLOE, Annotator, Config, DataLoader};

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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // config
    let config = Config::yoloe_v8s_seg_pf()
        // yoloe_v8m_seg_pf()
        // yoloe_v8l_seg_pf()
        // yoloe_11s_seg_pf()
        // yoloe_11m_seg_pf()
        // yoloe_11l_seg_pf()
        .with_model_dtype(args.dtype.as_str().parse()?)
        .with_model_device(args.device.as_str().parse()?)
        .commit()?;
    let mut model = YOLOE::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);

    // annotate
    let annotator = Annotator::default()
        .with_mask_style(usls::MaskStyle::default().with_draw_polygon_largest(true));

    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", "YOLOE-prompt-free", model.spec()])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }
    usls::perf(false);

    Ok(())
}
