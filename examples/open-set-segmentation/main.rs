use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{Sam3Image, Sam3Prompt, YOLOEPromptBased},
    Annotator, Config, DataLoader, Model, Source, YOLOEPrompt,
};

mod sam3_image;
#[path = "../utils/mod.rs"]
mod utils;
mod yoloe_prompt_based;

#[derive(Parser)]
#[command(author, version, about = "Open-Set Segmentation Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, default_value = "0.5")]
    pub confs: Vec<f32>,

    /// Prompts: "text", "text;pos:x,y", etc.
    #[arg(short = 'p', long, global = true, default_value = "person")]
    pub prompts: Vec<String>,

    #[command(subcommand)]
    pub command: Commands,

    /// Whether to cutout the annotated region
    #[arg(long, global = true, default_value = "true")]
    pub cutout: bool,
}

#[derive(Subcommand)]
enum Commands {
    YOLOEPromptBased(yoloe_prompt_based::YoloePromptArgs),
    Sam3Image(sam3_image::Sam3ImageArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    let annotator = Annotator::default()
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(true)
                .with_cutout(cli.cutout)
                .with_draw_polygon_largest(true),
        )
        .with_polygon_style(usls::PolygonStyle::default().with_thickness(2));

    match &cli.command {
        Commands::Sam3Image(args) => {
            let config = sam3_image::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            run_sam3_image(config, cli.source, &annotator, args, &cli.prompts)?
        }
        Commands::YOLOEPromptBased(args) => {
            let config = yoloe_prompt_based::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;

            run_yoloe_prompt_based(config, cli.source, &annotator, args, &cli.prompts)?
        }
    }

    usls::perf(false);

    Ok(())
}

fn run_yoloe_prompt_based(
    config: Config,
    source: Source,
    annotator: &Annotator,
    args: &yoloe_prompt_based::YoloePromptArgs,
    prompts: &[String],
) -> Result<()> {
    if prompts.is_empty() {
        anyhow::bail!("No prompt. Use -p \"class_name\" or -p \"xyxy:x1,y1,x2,y2,class_name\"");
    }

    let prompt = YOLOEPrompt::parse(prompts, args.prompt_image.as_deref())?;
    let mut model = YOLOEPromptBased::new(config)?;

    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    // Draw visual prompt boxes on the prompt image if visual prompt is used
    if prompt.is_visual() {
        prompt.draw(annotator)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs/open-set-segmentation", "YOLOE-prompt", &model.spec])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }

    for xs in &dl {
        let ys = model.forward((&xs, &prompt))?;
        tracing::info!("ys: {ys:?}");

        for (x, y) in xs.iter().zip(ys.iter()) {
            if y.is_empty() {
                continue;
            }

            let annotated = annotator.annotate(x, y)?;
            annotated.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&[
                        "runs/open-set-segmentation",
                        "YOLOE-prompt",
                        &model.spec
                    ])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?;
        }
    }
    Ok(())
}

fn run_sam3_image(
    config: Config,
    source: Source,
    annotator: &Annotator,
    args: &sam3_image::Sam3ImageArgs,
    prompts: &[String],
) -> Result<()> {
    if prompts.is_empty() {
        anyhow::bail!("No prompt. Use -p \"text\" or -p \"text;pos:x,y,w,h\"");
    }
    let prompts: Vec<Sam3Prompt> = prompts
        .iter()
        .map(|s| s.parse())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut model = Sam3Image::new(config)?;
    let dl = DataLoader::new(source)?
        .with_batch(args.visual_encoder_batch)
        .with_progress_bar(true)
        .stream()?;

    for batch in dl {
        let ys = model.forward((&batch, &prompts))?;
        tracing::info!("ys: {:?}", ys);
        for (img, y) in batch.iter().zip(ys.iter()) {
            let mut annotated = annotator.annotate(img, y)?;
            for prompt in &prompts {
                annotated = annotator.annotate(&annotated, &prompt.boxes)?;
            }
            annotated.save(
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/open-set-segmentation", "sam3-image"])?
                    .join(format!("{}.jpg", usls::timestamp(None))),
            )?;
        }
    }
    Ok(())
}
