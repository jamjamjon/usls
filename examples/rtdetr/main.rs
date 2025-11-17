use anyhow::Result;
use usls::{models::RTDETR, Annotator, Config, DataLoader, Scale, Version};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu\")")]
    device: String,

    /// scale
    #[argh(option, default = "String::from(\"s\")")]
    scale: String,

    /// dtype
    #[argh(option, default = "String::from(\"q4f16\")")]
    dtype: String,

    /// model
    #[argh(option)]
    model: Option<String>,

    /// version
    #[argh(option, default = "1.0")]
    ver: f32,

    /// confidences
    #[argh(option)]
    confs: Vec<f32>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // config
    let config = match args.model {
        Some(model) => Config::rtdetr().with_model_file(&model),
        None => {
            match args.ver.try_into()? {
                Version(1, 0, _) => match args.scale.parse()? {
                    Scale::Named(ref name) if name == "r18" => Config::rtdetr_v1_r18(),
                    Scale::Named(ref name) if name == "r18-obj365" => Config::rtdetr_v1_r18_obj365(),
                    Scale::Named(ref name) if name == "r34" => Config::rtdetr_v1_r34(),
                    Scale::Named(ref name) if name == "r50" => Config::rtdetr_v1_r50(),
                    Scale::Named(ref name) if name == "r50-obj365" => Config::rtdetr_v1_r50_obj365(),
                    Scale::Named(ref name) if name == "r101" => Config::rtdetr_v1_r101(),
                    Scale::Named(ref name) if name == "r101-obj365" => Config::rtdetr_v1_r101_obj365(),
                    _ => unimplemented!(
                        "Unsupported model scale: {:?} for RT-DETRv1. Try r18, r18-obj365, r34, r50, r50-obj365, r101, r101-obj365.",
                        args.scale,
                    ),
                },
                Version(2, 0, _) => match args.scale.parse()?{
                    Scale::S => Config::rtdetr_v2_s(),
                    Scale::M => Config::rtdetr_v2_m(),
                    Scale::Named(ref name) if name == "ms" => Config::rtdetr_v2_ms(),
                    Scale::L => Config::rtdetr_v2_l(),
                    Scale::X => Config::rtdetr_v2_x(),
                    _ => unimplemented!(
                        "Unsupported model scale: {:?} for RT-DETRv2. Try s, m, ms, l, x.",
                        args.scale,
                    ),
                },
                Version(4, 0, _) => match args.scale.parse()?{
                    Scale::S => Config::rtdetr_v4_s(),
                    Scale::M => Config::rtdetr_v4_m(),
                    Scale::L => Config::rtdetr_v4_l(),
                    Scale::X => Config::rtdetr_v4_x(),
                    _ => unimplemented!(
                        "Unsupported model scale: {:?} for RT-DETRv4. Try s, m, l, x.",
                        args.scale,
                    ),
                },
                _ => unimplemented!(
                    "Unsupported model version: {:?}. Try v1, v2, v4 for RT-DETR.",
                    args.ver
                ),
            }
        }
    }
    .with_dtype_all(args.dtype.parse()?)
    .with_device_all(args.device.parse()?)
    .with_class_confs(if args.confs.is_empty() {
            &[0.35]
        } else {
            &args.confs
        })
    .commit()?;
    let mut model = RTDETR::new(config)?;

    // load
    let xs = DataLoader::try_read_n(&["./assets/bus.jpg"])?;

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

    // summary
    usls::perf(false);

    Ok(())
}
