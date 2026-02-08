use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{DWPose, RTMPose, RTMO, YOLO},
    Annotator, Config, DataLoader, Model, Scale, Source, Y,
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
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    #[arg(long, global = true, value_delimiter = ',', default_values_t = vec![0.5])]
    pub confs: Vec<f32>,

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
    let annotator = Annotator::default().with_keypoint_style(
        usls::KeypointStyle::default()
            .with_skeleton(usls::SKELETON_COCO_19.into())
            .with_text_visible(false)
            .show_confidence(false)
            .show_id(true)
            .show_name(true),
    );
    let yolo_config = Config::yolo_detect()
        .with_scale(Scale::N)
        .with_model_dtype(usls::DType::Fp32)
        .with_version(26.into())
        .retain_classes(&[0])
        .with_model_batch_size(1)
        .with_class_confs(if cli.confs.is_empty() {
            &[0.5]
        } else {
            &cli.confs
        });

    match &cli.command {
        Commands::Rtmo(args) => {
            let config = rtmo::config(args)?.commit()?;
            let mut model = RTMO::new(config)?;
            let dl = DataLoader::new(&cli.source)?
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
        Commands::Dwpose(args) => {
            let yolo_config = yolo_config
                .with_model_device(args.device)
                .with_image_processor_device(args.processor_device)
                .commit()?;
            let pose_config = dwpose::config(args)?.commit()?;
            run_with_detector::<YOLO, DWPose>(
                yolo_config,
                pose_config,
                &cli.source,
                "dwpose",
                &annotator.with_keypoint_style(
                    usls::KeypointStyle::default()
                        .with_radius(1)
                        .with_skeleton(
                            (usls::SKELETON_COCO_65, usls::SKELETON_COLOR_COCO_65).into(),
                        )
                        .show_id(false)
                        .show_confidence(false)
                        .show_name(false),
                ),
            )
        }
        Commands::Rtmpose(args) => {
            let yolo_config = yolo_config
                .with_model_device(args.device)
                .with_image_processor_device(args.processor_device)
                .commit()?;
            let pose_config = rtmpose::config(args)?.commit()?;
            let is_coco = args.is_coco;
            run_with_detector::<YOLO, RTMPose>(
                yolo_config,
                pose_config,
                &cli.source,
                "rtmpose",
                &annotator
                    .with_hbb_style(usls::HbbStyle::default().with_draw_fill(true))
                    .with_keypoint_style(
                        usls::KeypointStyle::default()
                            .with_radius(2)
                            .with_skeleton(if is_coco {
                                (usls::SKELETON_COCO_19, usls::SKELETON_COLOR_COCO_19).into()
                            } else {
                                (usls::SKELETON_HALPE_27, usls::SKELETON_COLOR_HALPE_27).into()
                            })
                            .show_id(false)
                            .show_confidence(false)
                            .show_name(false),
                    ),
            )
        }
        Commands::Rtmw(args) => {
            let yolo_config = yolo_config
                .with_model_device(args.device)
                .with_image_processor_device(args.processor_device)
                .commit()?;
            let pose_config = rtmw::config(args)?.commit()?;
            run_with_detector::<YOLO, RTMPose>(
                yolo_config,
                pose_config,
                &cli.source,
                "rtmw",
                &annotator
                    .with_hbb_style(usls::HbbStyle::default().with_draw_fill(true))
                    .with_keypoint_style(
                        usls::KeypointStyle::default()
                            .with_radius(1)
                            .with_skeleton(
                                (usls::SKELETON_COCO_65, usls::SKELETON_COLOR_COCO_65).into(),
                            )
                            .show_id(false)
                            .show_confidence(false)
                            .show_name(false),
                    ),
            )
        }
    }?;

    usls::perf_chart();
    Ok(())
}

fn run_with_detector<D, E>(
    detector_config: Config,
    estimator_config: Config,
    source: &Source,
    model_name: &str,
    annotator: &Annotator,
) -> Result<()>
where
    for<'a> D: Model<Input<'a> = &'a [usls::Image]>,
    for<'a> E: Model<Input<'a> = &'a [usls::Image]>,
{
    // Build models
    let mut detector = D::new(detector_config)?;
    let mut pose_model = E::new(estimator_config)?;
    let spec = pose_model.spec().to_string();

    // Build dataloader
    let dl = DataLoader::new(source)?
        .with_batch(1)
        .with_progress_bar(true)
        .stream()?;

    for xs in &dl {
        let ys_det = detector.forward(&xs)?;

        for (img, y_det) in xs.iter().zip(ys_det.iter()) {
            let hbbs = y_det.hbbs();
            if hbbs.is_empty() {
                continue;
            }

            // Process in batches
            let mut all_keypointss = Vec::new();
            for (chunk_idx, hbb_chunk) in hbbs.chunks(pose_model.batch()).enumerate() {
                // Crop regions
                let chunk_crops = img.crop(hbb_chunk)?;

                // Forward pass
                let chunk_ys_pose = pose_model.forward(chunk_crops.as_slice())?;

                // Process results and map back to original coordinates
                for (result_idx, y) in chunk_ys_pose.iter().enumerate() {
                    let original_hbb_idx = chunk_idx * pose_model.batch() + result_idx;
                    if original_hbb_idx >= hbbs.len() {
                        continue;
                    }

                    let hbb = &hbbs[original_hbb_idx];
                    for kpts in &y.keypointss {
                        // Map keypoints back to original image coordinates
                        let mapped_kpts: Vec<_> = kpts
                            .iter()
                            .map(|k| k.clone().with_xy(k.x() + hbb.xmin(), k.y() + hbb.ymin()))
                            .collect();
                        all_keypointss.push(mapped_kpts);
                    }
                }
            }

            let y = Y::default().with_keypointss(&all_keypointss);
            annotator.annotate(img, &y)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/pose-estimation", model_name, &spec])?
                    .join(usls::timestamp(None))
                    .display()
            ))?
        }
    }

    Ok(())
}
