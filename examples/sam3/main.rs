use anyhow::Result;
use usls::{
    vlm::{Sam3Prompt, SAM3},
    Annotator, Config, DataLoader, Task,
};

#[derive(argh::FromArgs)]
/// SAM3 - Segment Anything Model 3
struct Args {
    /// task (sam3-image, sam3-tracker)
    #[argh(option, default = "String::from(\"sam3-image\")")]
    task: String,

    /// device (cpu:0, cuda:0, etc.)
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// source image paths (can specify multiple)
    #[argh(option, default = "vec![String::from(\"./assets/sam3-demo.jpg\")]")]
    source: Vec<String>,

    /// prompts: "text;pos:x,y,w,h;neg:x,y,w,h" (can specify multiple)
    #[argh(option, short = 'p')]
    prompt: Vec<String>,

    /// confidence threshold (default: 0.5)
    #[argh(option, default = "0.5")]
    conf: f32,

    /// show mask
    #[argh(option, default = "true")]
    show_mask: bool,

    /// dtype
    #[argh(option, default = "String::from(\"q4f16\")")]
    dtype: String,

    /// vision encoder batch min
    #[argh(option, default = "1")]
    vision_batch_min: usize,

    /// vision encoder batch opt
    #[argh(option, default = "1")]
    vision_batch: usize,

    /// vision encoder batch max
    #[argh(option, default = "8")]
    vision_batch_max: usize,

    /// text encoder batch min
    #[argh(option, default = "1")]
    text_batch_min: usize,

    /// text encoder batch opt
    #[argh(option, default = "4")]
    text_batch: usize,

    /// text encoder batch max
    #[argh(option, default = "16")]
    text_batch_max: usize,

    /// geometry encoder batch min
    #[argh(option, default = "1")]
    geo_batch_min: usize,

    /// geometry encoder batch opt
    #[argh(option, default = "8")]
    geo_batch: usize,

    /// geometry encoder batch max
    #[argh(option, default = "16")]
    geo_batch_max: usize,

    /// decoder batch min
    #[argh(option, default = "1")]
    decoder_batch_min: usize,

    /// decoder batch opt
    #[argh(option, default = "1")]
    decoder_batch: usize,

    /// decoder batch max
    #[argh(option, default = "8")]
    decoder_batch_max: usize,
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
    let config = match args.task.parse()? {
        Task::Sam3Image => Config::sam3_image()
            .with_textual_encoder_batch_min_opt_max(
                args.text_batch_min,
                args.text_batch,
                args.text_batch_max,
            )
            .with_encoder_batch_min_opt_max(args.geo_batch_min, args.geo_batch, args.geo_batch_max),
        Task::Sam3Tracker => Config::sam3_tracker(),
        _ => anyhow::bail!(
            "Sam3 Task now only support: {}, {}",
            Task::Sam3Image,
            Task::Sam3Tracker
        ),
    }
    .with_visual_encoder_batch_min_opt_max(
        args.vision_batch_min,
        args.vision_batch,
        args.vision_batch_max,
    )
    .with_decoder_batch_min_opt_max(
        args.decoder_batch_min,
        args.decoder_batch,
        args.decoder_batch_max,
    )
    .with_dtype_all(args.dtype.parse()?)
    .with_class_confs(&[args.conf])
    .with_device_all(args.device.parse()?)
    .commit()?;
    let mut model = SAM3::new(config)?;

    // Annotator for model results
    let annotator = Annotator::default()
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(args.show_mask)
                .with_cutout(true)
                .with_draw_polygon_largest(true),
        )
        .with_polygon_style(usls::PolygonStyle::default().with_thickness(2));

    // DataLoader with batch iteration
    let dataloader = DataLoader::from_paths(&args.source)?
        .with_batch(args.vision_batch)
        .with_progress_bar(true)
        .build()?;

    // Process in batches
    for batch in dataloader {
        let ys = model.forward(&batch, &prompts)?;
        println!("ys: {:?}", ys);

        for (img, y) in batch.iter().zip(ys.iter()) {
            // Annotate model results
            let mut annotated = annotator.annotate(img, y)?;

            // Draw prompts
            for prompt in &prompts {
                annotated = annotator.annotate(&annotated, &prompt.boxes)?;
                annotated = annotator.annotate(&annotated, &prompt.points)?;
            }

            annotated.save(
                usls::Dir::Current
                    .base_dir_with_subs(&["runs", model.spec()])?
                    .join(format!("{}.jpg", usls::timestamp(None))),
            )?;
        }
    }

    // Performance stats
    usls::perf(false);

    Ok(())
}
