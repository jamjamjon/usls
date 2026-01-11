use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{Sam3Image, Sam3Prompt, YOLOEPrompt},
    Annotator, Config, DataLoader, Hbb, Model, Source,
};

mod sam3_image;
#[path = "../utils/mod.rs"]
mod utils;
mod yoloe_prompt;

use crate::yoloe_prompt::Kind;

#[derive(Parser)]
#[command(author, version, about = "Open-Set Segmentation Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, value_delimiter = ',')]
    pub confs: Vec<f32>,

    /// Text prompts, labels (comma-separated)
    #[arg(
        long,
        global = true,
        value_delimiter = ',',
        default_value = "person,bus"
    )]
    pub labels: Vec<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    YOLOEPrompt(yoloe_prompt::YoloePromptArgs),
    Sam3Image(sam3_image::Sam3ImageArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    let annotator = Annotator::default()
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(true)
                .with_cutout(true)
                .with_draw_polygon_largest(true),
        )
        .with_polygon_style(usls::PolygonStyle::default().with_thickness(2));

    match &cli.command {
        Commands::Sam3Image(args) => {
            let config = sam3_image::config(args)?.commit()?;
            run_sam3_image(config, cli.source, &annotator, args)?
        }

        Commands::YOLOEPrompt(args) => {
            let config = yoloe_prompt::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            run_yoloe_prompt(config, cli.source, &annotator, args, &cli.labels)?
        }
    }
    usls::perf(false);

    Ok(())
}

fn run_yoloe_prompt(
    config: Config,
    source: Source,
    annotator: &Annotator,
    args: &yoloe_prompt::YoloePromptArgs,
    labels: &[String],
) -> Result<()> {
    let mut model = YOLOEPrompt::new(config)?;

    // Encode embedding
    let embedding = match args.kind {
        Kind::Visual => {
            let prompt_image = DataLoader::new("./assets/bus.jpg")?.try_read_one()?;
            model.encode_visual_prompt(
                prompt_image,
                &[Hbb::from_xyxy(221.52, 405.8, 344.98, 857.54).with_name("person")],
            )?
        }
        Kind::Textual => {
            model.encode_class_names(&labels.iter().map(|x| x.as_str()).collect::<Vec<_>>())?
        }
    };

    // Build dataloader
    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    // Run & Annotate
    for xs in &dl {
        let ys = model.run((&xs, &embedding))?;
        // println!("ys: {:?}", ys);

        for (x, y) in xs.iter().zip(ys.iter()) {
            if y.is_empty() {
                continue;
            }
            annotator.annotate(x, y)?.save(format!(
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
) -> Result<()> {
    if args.prompt.is_empty() {
        anyhow::bail!("No prompt. Use -p \"text\" or -p \"text;pos:x,y,w,h\"");
    }
    let prompts: Vec<Sam3Prompt> = args
        .prompt
        .iter()
        .map(|s| s.parse())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut model = Sam3Image::new(config)?;
    let dl = DataLoader::new(source)?
        .with_batch(args.vision_batch)
        .with_progress_bar(true)
        .stream()?;

    for batch in dl {
        let ys = model.forward((&batch, &prompts))?;
        // println!("ys: {:?}", ys);
        for (img, y) in batch.iter().zip(ys.iter()) {
            let mut annotated = annotator.annotate(img, y)?;
            for prompt in &prompts {
                annotated = annotator.annotate(&annotated, &prompt.boxes)?;
                annotated = annotator.annotate(&annotated, &prompt.points)?;
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
