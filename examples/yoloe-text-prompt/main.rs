use anyhow::Result;
use usls::{models::YOLO, Annotator, Config, DataLoader, Style};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// source: image, image folder, video stream
    #[argh(option, default = "String::from(\"./assets/bus.jpg\")")]
    source: String,

    /// dtype
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,

    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// open class names
    #[argh(
        option,
        default = "vec![
            String::from(\"person\"),
            String::from(\"bus\"),
            String::from(\"dog\"),
            String::from(\"cat\"),
            String::from(\"sign\"),
            String::from(\"tree\"),
        ]"
    )]
    labels: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // config
    let config = Config::yoloe_v8l_seg_tp_80()
        .with_model_dtype(args.dtype.as_str().parse()?)
        .with_model_device(args.device.as_str().parse()?)
        .with_batch_size_all(1)
        .commit()?;
    let mut model = YOLO::new(config)?;

    // build dataloader
    let dl = DataLoader::new(&args.source)?
        .with_batch(model.batch() as _)
        .build()?;

    // annotator
    let annotator = Annotator::default()
        .with_hbb_style(Style::hbb().with_draw_fill(true))
        .with_mask_style(Style::mask().with_draw_mask_polygon_largest(true));

    // encode text prompts
    let text_embeddings =
        model.encode_class_names(&args.labels.iter().map(|x| x.as_str()).collect::<Vec<_>>())?;

    // run & annotate
    for xs in &dl {
        // infer with text embeddings
        let ys = model.forward_with_te(&xs, &text_embeddings)?;
        println!("ys: {:?}", ys);

        for (x, y) in xs.iter().zip(ys.iter()) {
            if y.is_empty() {
                continue;
            }
            annotator.annotate(x, y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs", model.spec()])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?;
        }
    }

    usls::perf(false);

    Ok(())
}
