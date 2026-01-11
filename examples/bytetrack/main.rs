use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{RFDETR, YOLO},
    Annotator, ByteTracker, Config, DataLoader, Model, Source, Viewer, SKELETON_COCO_19,
    SKELETON_COLOR_COCO_19,
};

#[path = "../object-detection/rfdetr.rs"]
mod rfdetr;
#[path = "../utils/mod.rs"]
mod utils;
#[path = "../yolo/args.rs"]
mod yolo_args;

#[derive(Parser)]
#[command(author, version, about = "Multi-Object Tracking Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, required = true)]
    pub source: Source,

    /// Retain classes
    #[arg(long, value_delimiter = ',')]
    pub retain_classes: Vec<usize>,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, value_delimiter = ',')]
    pub confs: Vec<f32>,

    #[arg(long, global = true, default_value = "false")]
    pub save: bool,

    #[arg(long, global = true, default_value = "1.0")]
    pub window_scale: f32,

    #[arg(long, global = true, default_value = "1")]
    pub delay: u64,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    YoloTrack(yolo_args::YoloArgs),
    RfdetrTrack(rfdetr::RfdetrArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    let annotator = Annotator::default()
        .with_obb_style(usls::ObbStyle::default().with_draw_fill(true))
        .with_hbb_style(
            usls::HbbStyle::default()
                .with_draw_fill(true)
                .with_palette(&usls::Color::palette_coco_80()),
        )
        .with_keypoint_style(
            usls::KeypointStyle::default()
                .with_skeleton((SKELETON_COCO_19, SKELETON_COLOR_COCO_19).into())
                .show_confidence(false)
                .show_id(true)
                .show_name(false),
        )
        .with_mask_style(usls::MaskStyle::default().with_draw_polygon_largest(true));

    match &cli.command {
        Commands::YoloTrack(args) => {
            let config = yolo_args::config(args)?
                .with_class_confs(if cli.confs.is_empty() {
                    &[0.1]
                } else {
                    &cli.confs
                })
                .retain_classes(&cli.retain_classes)
                .commit()?;
            run::<YOLO>(config, &cli, &annotator)
        }
        Commands::RfdetrTrack(args) => {
            let config = rfdetr::config(args)?
                .with_class_confs(if cli.confs.is_empty() {
                    &[0.1]
                } else {
                    &cli.confs
                })
                .retain_classes(&cli.retain_classes)
                .commit()?;
            run::<RFDETR>(config, &cli, &annotator)
        }
    }?;

    usls::perf(false);
    Ok(())
}

fn run<M>(config: Config, cli: &Cli, annotator: &Annotator) -> Result<()>
where
    for<'a> M: Model<Input<'a> = &'a [usls::Image]>,
{
    let mut model = M::new(config)?;
    let dl = DataLoader::new(&cli.source)?
        .with_batch(1) // use batch 1 for tracking
        .with_progress_bar(true)
        .stream()?;

    let mut viewer = Viewer::default().with_window_scale(cli.window_scale);
    let mut tracker = ByteTracker::default().with_max_age(60);

    for xs in &dl {
        // check window
        if viewer.is_window_exist_and_closed() {
            break;
        }

        // inference
        let ys = model.forward(&xs)?;

        // first frame
        let frame = &xs[0];

        // get hbbs
        let hbbs = ys[0].hbbs();
        if hbbs.is_empty() {
            // imshow
            viewer.imshow(frame)?;
        } else {
            // track & annotate & imshow
            let tracked_hbbs = tracker.update(hbbs)?;
            let frame = annotator.annotate(frame, &tracked_hbbs)?;
            viewer.imshow(&frame)?;
        };

        // wait key
        if let Some(key) = viewer.wait_key(cli.delay) {
            if key == usls::Key::Escape {
                break;
            }
        }

        // save
        if cli.save {
            viewer.write_video_frame(frame)?;
        }
    }

    Ok(())
}
