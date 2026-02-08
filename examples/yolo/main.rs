use anyhow::Result;
use clap::Parser;
use usls::{
    models::YOLO, Annotator, DataLoader, Model, Source, SKELETON_COCO_19, SKELETON_COLOR_COCO_19,
};

mod args;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser, Debug)]
#[command(author, version, about = "YOLO Example", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image, image folder, video stream
    #[arg(long, default_value = "./assets/bus.jpg")]
    source: Source,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, value_delimiter = ',', default_values_t = vec![0.35, 0.3])]
    pub confs: Vec<f32>,

    /// Exclude classes
    #[arg(long, value_delimiter = ',')]
    pub exclude_classes: Vec<usize>,

    /// Retain classes
    #[arg(long, value_delimiter = ',')]
    pub retain_classes: Vec<usize>,

    #[command(flatten)]
    pub args: args::YoloArgs,
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    let config = args::config(&cli.args)?
        .with_class_confs(&cli.confs)
        .retain_classes(&cli.retain_classes)
        .exclude_classes(&cli.exclude_classes)
        .commit()?;

    // build model
    let mut model = YOLO::new(config)?;

    // build dataloader
    let dl = DataLoader::new(&cli.source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    // build annotator
    let annotator = Annotator::default()
        .with_hbb_style(usls::HbbStyle::default().with_palette(&usls::Color::palette_coco_80()))
        .with_keypoint_style(
            usls::KeypointStyle::default()
                .with_text_visible(true)
                .with_skeleton((SKELETON_COCO_19, SKELETON_COLOR_COCO_19).into())
                .show_confidence(false)
                .show_name(true),
        )
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(true)
                .with_cutout(true)
                .with_draw_polygon_largest(true),
        );

    // run & annotate
    for xs in &dl {
        let ys = model.forward(&xs)?;
        // println!("ys: {ys:?}");

        for (x, y) in xs.iter().zip(ys.iter()) {
            annotator.annotate(x, y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs", model.spec()])?
                    .join(usls::timestamp(None))
                    .display(),
            ))?;
        }
    }

    usls::perf_chart();

    Ok(())
}
