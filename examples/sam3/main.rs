use anyhow::Result;
use usls::{
    models::{Sam3Prompt, SAM3},
    Annotator, Config, DataLoader,
};

#[derive(argh::FromArgs)]
/// SAM3 - Segment Anything Model 3
struct Args {
    /// device (cpu:0, cuda:0, etc.)
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image paths (can specify multiple)
    #[argh(
        option,
        default = "vec![
        String::from(\"./assets/sam3-demo.jpg\"),
        // String::from(\"./assets/bus.jpg\")
    ]"
    )]
    source: Vec<String>,

    /// prompts: "text;pos:x,y,w,h;neg:x,y,w,h" (can specify multiple)
    #[argh(option, short = 'p')]
    prompt: Vec<String>,

    /// confidence threshold (default: 0.5)
    #[argh(option, default = "0.5")]
    conf: f32,

    /// batch size min (default: 1)
    #[argh(option, default = "1")]
    batch_min: usize,

    /// batch size (default: 1)
    #[argh(option, default = "1")]
    batch: usize,

    /// batch size max (default: 4)
    #[argh(option, default = "4")]
    batch_max: usize,

    /// dtype
    #[argh(option, default = "String::from(\"q4f16\")")]
    dtype: String,

    /// show mask
    #[argh(switch)]
    show_mask: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();

    // Parse prompts
    if args.prompt.is_empty() {
        anyhow::bail!("No prompt. Use -p \"text\" or -p \"visual;pos:x,y,w,h\"");
    }
    let prompts: Vec<Sam3Prompt> = args
        .prompt
        .iter()
        .map(|s| s.parse())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Build model
    let config = Config::sam3_image_predictor()
        .with_batch_size_all_min_opt_max(args.batch_min, args.batch, args.batch_max)
        .with_device_all(args.device.parse()?)
        .with_dtype_all(args.dtype.parse()?)
        .with_class_confs(&[args.conf])
        .with_num_dry_run_all(1)
        .commit()?;
    let mut model = SAM3::new(config)?;

    // Annotator
    let annotator = Annotator::default().with_mask_style(
        usls::Style::mask()
            .with_draw_mask_polygon_largest(true)
            .with_visible(args.show_mask),
    );
    let output_dir = usls::Dir::Current.base_dir_with_subs(&["runs", model.spec()])?;

    // DataLoader with batch iteration
    let dataloader = DataLoader::from_paths(&args.source)?
        .with_batch(args.batch)
        .with_progress_bar(true)
        .build()?;

    // Process in batches
    for batch in dataloader {
        let ys = model.forward(&batch, &prompts)?;
        println!("ys: {:?}", ys);

        for (img, y) in batch.iter().zip(ys.iter()) {
            annotator
                .annotate(img, y)?
                .save(output_dir.join(format!("{}.jpg", usls::timestamp(None))))?;
        }
    }

    usls::perf(false);
    Ok(())
}
