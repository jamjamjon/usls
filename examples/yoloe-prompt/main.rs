use anyhow::Result;
use usls::{models::vlm::YOLOE, Annotator, Config, DataLoader, Hbb};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// source: image, image folder, video stream
    #[argh(option, default = "String::from(\"./assets/bus.jpg\")")]
    source: String,

    /// dtype
    #[argh(option, default = "String::from(\"fp32\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// open class names
    #[argh(
        option,
        default = "vec![
            String::from(\"person\"),
            String::from(\"dog\"),
            String::from(\"bus\"),
            String::from(\"cat\"),
            String::from(\"sign\"),
            String::from(\"tree\"),
        ]"
    )]
    labels: Vec<String>,

    /// batch size
    #[argh(option, default = "1")]
    batch_size: usize,

    /// visual or textual
    #[argh(option, default = "true")]
    visual: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // config
    let config = if args.visual {
        Config::yoloe_11s_seg_vp()
    } else {
        Config::yoloe_v8s_seg_tp().with_textual_encoder_dtype("fp16".parse()?) // Use FP32 when TensorRT is enabled
    }
    .with_batch_size_all_min_opt_max(1, args.batch_size, 8)
    .with_model_dtype(args.dtype.as_str().parse()?)
    .with_device_all(args.device.as_str().parse()?)
    .with_class_confs(&[0.25])
    .commit()?;
    let mut model = YOLOE::new(config)?;

    // encode visual or textual
    let embedding = if args.visual {
        let prompt_image = DataLoader::try_read_one("./assets/bus.jpg")?;
        model.encode_visual_prompt(
            prompt_image,
            &[
                Hbb::from_xyxy(221.52, 405.8, 344.98, 857.54).with_name("person"),
                // Hbb::from_xyxy(120., 425., 160., 445.).with_name("glasses"), // TODO
            ],
        )?
    } else {
        model.encode_class_names(&args.labels.iter().map(|x| x.as_str()).collect::<Vec<_>>())?
    };

    // build dataloader
    let dl = DataLoader::new(&args.source)?
        .with_batch(model.batch() as _)
        .build()?;

    // annotator
    let annotator = Annotator::default()
        .with_mask_style(usls::MaskStyle::default().with_draw_polygon_largest(true));

    // run & annotate
    for xs in &dl {
        let ys = model.forward_with_embedding(&xs, &embedding)?;
        println!("ys: {:?}", ys);

        for (x, y) in xs.iter().zip(ys.iter()) {
            if y.is_empty() {
                continue;
            }
            annotator.annotate(x, y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs", "YOLOE-prompt", model.spec()])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?;
        }
    }

    usls::perf(false);

    Ok(())
}
