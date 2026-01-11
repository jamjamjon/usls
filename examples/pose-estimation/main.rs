use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{DWPose, RTMPose, RTMO, YOLO},
    Annotator, Config, DataLoader, Model, Scale, Source,
};

mod dwpose;
mod rtmo;
mod rtmpose;
mod rtmw;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Pose Estimation Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Dwpose(dwpose::DwposeArgs),
    Rtmo(rtmo::RtmoArgs),
    Rtmpose(rtmpose::RtmposeArgs),
    Rtmw(rtmw::RtmwArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    match &cli.command {
        Commands::Dwpose(args) => {
            let config = dwpose::config(args)?.commit()?;
            run_dwpose(config, &cli.source, args)
        }
        Commands::Rtmo(args) => {
            let config = rtmo::config(args)?.commit()?;
            run_rtmo(config, &cli.source, args)
        }
        Commands::Rtmpose(args) => {
            let config = rtmpose::config(args)?.commit()?;
            run_rtmpose(config, &cli.source, args)
        }
        Commands::Rtmw(args) => {
            let config = rtmw::config(args)?.commit()?;
            run_rtmw(config, &cli.source, args)
        }
    }?;

    usls::perf(false);
    Ok(())
}

fn run_dwpose(config: Config, source: &Source, args: &dwpose::DwposeArgs) -> Result<()> {
    let yolo_config = Config::yolo_detect()
        .with_scale(Scale::N)
        .with_version(8.into())
        .with_model_device(args.device)
        .retain_classes(&[0])
        .with_class_confs(&[0.5])
        .commit()?;
    let mut yolo = YOLO::new(yolo_config)?;

    let mut dwpose = DWPose::new(config)?;

    let annotator = Annotator::default().with_keypoint_style(
        usls::KeypointStyle::default()
            .with_radius(2)
            .with_skeleton((usls::SKELETON_COCO_65, usls::SKELETON_COLOR_COCO_65).into())
            .show_id(false)
            .show_confidence(false)
            .show_name(false),
    );

    let dl = DataLoader::new(source)?.with_batch(1).stream()?;

    for xs in &dl {
        let ys_det = yolo.forward(&xs)?;

        for (x, y_det) in xs.iter().zip(ys_det.iter()) {
            let y = dwpose.run_with_bboxes(x, Some(y_det.hbbs()))?;

            annotator.annotate(x, &y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/pose-estimation", "dwpose", dwpose.spec()])?
                    .join(usls::timestamp(None))
                    .display()
            ))?
        }
    }

    Ok(())
}

fn run_rtmo(config: Config, source: &Source, _args: &rtmo::RtmoArgs) -> Result<()> {
    let mut model = RTMO::new(config)?;

    let annotator = Annotator::default().with_keypoint_style(
        usls::KeypointStyle::default()
            .with_skeleton(usls::SKELETON_COCO_19.into())
            .show_confidence(false)
            .show_id(true)
            .show_name(true),
    );

    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &dl {
        let ys = model.forward(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/pose-estimation", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }

    Ok(())
}

fn run_rtmpose(config: Config, source: &Source, args: &rtmpose::RtmposeArgs) -> Result<()> {
    let yolo_config = Config::yolo_detect()
        .with_scale(Scale::N)
        .with_version(8.into())
        .with_model_device(args.device)
        .retain_classes(&[0])
        .with_class_confs(&[0.5])
        .commit()?;
    let mut yolo = YOLO::new(yolo_config)?;

    let mut rtmpose = RTMPose::new(config)?;

    let annotator = Annotator::default()
        .with_hbb_style(usls::HbbStyle::default().with_draw_fill(true))
        .with_keypoint_style(
            usls::KeypointStyle::default()
                .with_radius(4)
                .with_skeleton(if args.is_coco {
                    (usls::SKELETON_COCO_19, usls::SKELETON_COLOR_COCO_19).into()
                } else {
                    (usls::SKELETON_HALPE_27, usls::SKELETON_COLOR_HALPE_27).into()
                })
                .show_id(false)
                .show_confidence(false)
                .show_name(false),
        );

    let dl = DataLoader::new(source)?.with_batch(1).stream()?;

    for xs in &dl {
        let ys_det = yolo.forward(&xs)?;

        for (x, y_det) in xs.iter().zip(ys_det.iter()) {
            let y = rtmpose.run_with_bboxes(x, Some(y_det.hbbs()))?;

            annotator.annotate(x, &y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/pose-estimation", rtmpose.spec()])?
                    .join(usls::timestamp(None))
                    .display()
            ))?
        }
    }

    Ok(())
}

fn run_rtmw(config: Config, source: &Source, args: &rtmw::RtmwArgs) -> Result<()> {
    let yolo_config = Config::yolo_detect()
        .with_scale(Scale::N)
        .with_version(8.into())
        .with_model_device(args.device)
        .retain_classes(&[0])
        .with_class_confs(&[0.5])
        .commit()?;
    let mut yolo = YOLO::new(yolo_config)?;

    let mut rtmpose = RTMPose::new(config)?;

    let annotator = Annotator::default()
        .with_hbb_style(usls::HbbStyle::default().with_draw_fill(true))
        .with_keypoint_style(
            usls::KeypointStyle::default()
                .with_radius(2)
                .with_skeleton((usls::SKELETON_COCO_65, usls::SKELETON_COLOR_COCO_65).into())
                .show_id(false)
                .show_confidence(false)
                .show_name(false),
        );

    let dl = DataLoader::new(source)?.with_batch(1).stream()?;

    for xs in &dl {
        let ys_det = yolo.forward(&xs)?;

        for (x, y_det) in xs.iter().zip(ys_det.iter()) {
            let y = rtmpose.run_with_bboxes(x, Some(y_det.hbbs()))?;

            annotator.annotate(x, &y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/pose-estimation", rtmpose.spec()])?
                    .join(usls::timestamp(None))
                    .display()
            ))?
        }
    }

    Ok(())
}
