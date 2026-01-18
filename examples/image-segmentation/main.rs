use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{
        FastSAM, Sam3Prompt, Sam3Tracker, SamPrompt, YOLOEPromptFree, YOLOPv2, RFDETR, SAM, SAM2,
    },
    Annotator, Config, DataLoader, Model, Source,
};

mod fastsam;
mod rfdetr;
mod sam;
mod sam2;
mod sam3_tracker;
#[path = "../utils/mod.rs"]
mod utils;
mod yoloe_prompt_free;
mod yolop;

#[derive(Parser)]
#[command(author, version, about = "Object Detection Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, value_delimiter = ',', default_values_t = vec![0.5])]
    pub confs: Vec<f32>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Rfdetr(rfdetr::RfdetrArgs),
    YoloePromptFree(yoloe_prompt_free::YoloePromptFreeArgs),
    Sam(sam::SamArgs),
    Sam2(sam2::Sam2Args),
    Yolop(yolop::YolopArgs),
    Fastsam(fastsam::FastsamArgs),
    Sam3Tracker(sam3_tracker::Sam3TrackerArgs),
}

fn run<M>(config: Config, source: &Source, annotator: &Annotator) -> Result<()>
where
    for<'a> M: Model<Input<'a> = &'a [usls::Image]>,
{
    let mut model = M::new(config)?;
    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &dl {
        let ys = model.forward(&xs)?;
        // println!("{:?}", ys);
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/image-segmentation", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    let annotator = Annotator::default().with_mask_style(
        usls::MaskStyle::default()
            .with_visible(true)
            .with_cutout(true)
            .with_draw_polygon_largest(true),
    );

    match &cli.command {
        Commands::Rfdetr(args) => {
            let config = rfdetr::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            run::<RFDETR>(config, &cli.source, &annotator)
        }
        Commands::YoloePromptFree(args) => {
            let config = yoloe_prompt_free::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            let annotator = annotator
                .with_mask_style(usls::MaskStyle::default().with_draw_polygon_largest(true));
            run::<YOLOEPromptFree>(config, &cli.source, &annotator)
        }
        Commands::Sam(args) => {
            let config = sam::config(args)?.commit()?;
            run_sam(config, &cli.source)
        }
        Commands::Sam2(args) => {
            let config = sam2::config(args)?.commit()?;
            run_sam2(config, &cli.source)
        }
        Commands::Yolop(args) => {
            let config = yolop::config(args)?.commit()?;
            let annotator = annotator.with_polygon_style(
                usls::PolygonStyle::default()
                    .with_text_visible(true)
                    .show_name(true)
                    .show_id(true),
            );
            run::<YOLOPv2>(config, &cli.source, &annotator)
        }
        Commands::Fastsam(args) => {
            let config = fastsam::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            let annotator = annotator
                .with_hbb_style(
                    usls::HbbStyle::default()
                        .show_confidence(true)
                        .show_id(false)
                        .show_name(false),
                )
                .with_polygon_style(usls::PolygonStyle::default().with_thickness(2));
            run::<FastSAM>(config, &cli.source, &annotator)
        }
        Commands::Sam3Tracker(args) => {
            let config = sam3_tracker::config(args)?.commit()?;
            run_sam3_tracker(config, &cli.source, args)
        }
    }?;
    usls::perf(false);

    Ok(())
}

fn run_sam(config: Config, source: &Source) -> Result<()> {
    let mut model = SAM::new(config)?;
    let dl = DataLoader::new(source)?
        .with_batch(model.batch as _)
        .stream()?;

    let prompts = vec![SamPrompt::default()
        .with_xyxy(425., 600., 700., 875.)
        .with_negative_point(575., 750.)];

    let annotator = Annotator::default();
    for xs in &dl {
        let ys = model.run((&xs, &prompts))?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            annotator.annotate(x, y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/image-segmentation", model.spec()])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?;
        }
    }
    Ok(())
}

fn run_sam2(config: Config, source: &Source) -> Result<()> {
    let mut model = SAM2::new(config)?;
    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .stream()?;

    let prompts = vec![SamPrompt::default()
        .with_xyxy(75., 275., 1725., 850.)
        .with_xyxy(425., 600., 700., 875.)
        .with_xyxy(1375., 550., 1650., 800.)
        .with_xyxy(1240., 675., 1400., 750.)];

    let annotator = Annotator::default()
        .with_mask_style(usls::MaskStyle::default().with_draw_polygon_largest(true));

    for xs in &dl {
        let ys = model.forward((&xs, &prompts))?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            annotator.annotate(x, y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/image-segmentation", model.spec()])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?;
        }
    }
    Ok(())
}

fn run_sam3_tracker(
    config: Config,
    source: &Source,
    args: &sam3_tracker::Sam3TrackerArgs,
) -> Result<()> {
    if args.prompt.is_empty() {
        anyhow::bail!("No prompt. Use -p \"name;point:x,y,1\" or -p \"name;box:x,y,w,h\"");
    }
    let prompts: Vec<Sam3Prompt> = args
        .prompt
        .iter()
        .map(|s| s.parse())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut model = Sam3Tracker::new(config)?;
    let annotator = Annotator::default()
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(args.show_mask)
                .with_cutout(true)
                .with_draw_polygon_largest(true),
        )
        .with_polygon_style(usls::PolygonStyle::default().with_thickness(2));

    let dataloader = DataLoader::new(source)?
        .with_batch(args.vision_batch)
        .with_progress_bar(true)
        .stream()?;

    for batch in dataloader {
        let ys = model.forward((&batch, &prompts))?;
        for (img, y) in batch.iter().zip(ys.iter()) {
            let mut annotated = annotator.annotate(img, y)?;
            for prompt in &prompts {
                annotated = annotator.annotate(&annotated, &prompt.boxes)?;
                annotated = annotator.annotate(&annotated, &prompt.points)?;
            }
            annotated.save(
                usls::Dir::Current
                    .base_dir_with_subs(&["runs", "sam3-tracker"])?
                    .join(format!("{}.jpg", usls::timestamp(None))),
            )?;
        }
    }
    Ok(())
}
