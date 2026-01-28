use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{MobileGaze, YOLO},
    Annotator, Config, DataLoader, Model, Source,
};

mod mobile_gaze;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Gaze Estimation Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/gaze.png")]
    pub source: Source,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, value_delimiter = ',', default_values_t = vec![0.5])]
    pub confs: Vec<f32>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    MobileGaze(mobile_gaze::MobileGazeArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    let annotator = Annotator::default()
        .with_hbb_style(usls::HbbStyle::default().with_text_visible(false))
        .with_keypoint_style(
            usls::KeypointStyle::default()
                .with_visible(false)
                .with_text_visible(false),
        ); // Disable Face landmarks

    match &cli.command {
        Commands::MobileGaze(args) => {
            let yolo_config = Config::yolo11n_widerface()
                .with_model_dtype(usls::DType::Fp32)
                .with_model_batch_size(1)
                .with_model_device(args.device)
                .with_image_processor_device(args.processor_device)
                .with_class_confs(if cli.confs.is_empty() {
                    &[0.5]
                } else {
                    &cli.confs
                })
                .commit()?;
            let mobile_gaze_config = mobile_gaze::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            run::<YOLO, MobileGaze>(yolo_config, mobile_gaze_config, &cli, &annotator)
        }
    }?;
    usls::perf(false);

    Ok(())
}

fn run<D, E>(
    detector_config: Config,
    estimator_config: Config,
    cli: &Cli,
    annotator: &Annotator,
) -> Result<()>
where
    for<'a> D: Model<Input<'a> = &'a [usls::Image]>,
    for<'a> E: Model<Input<'a> = &'a [usls::Image]>,
{
    // Build models
    let mut face_detector = D::new(detector_config)?;
    let mut gaze_estimator = E::new(estimator_config)?;

    // Build dataloader
    let dl = DataLoader::new(&cli.source)?
        .with_batch(1)
        .with_progress_bar(true)
        .stream()?;

    // Inference pipeline
    for xs in &dl {
        // Face detection
        let ys_faces = face_detector.forward(&xs)?;

        // Process each image
        for (img, faces) in xs.into_iter().zip(ys_faces.iter()) {
            let mut rgba = img.to_rgba8();

            // Get detected face bounding boxes
            let hbbs = faces.hbbs();
            for (chunk_idx, hbb_chunk) in hbbs.chunks(gaze_estimator.batch()).enumerate() {
                // Crop face region from original image
                let face_crops = img.crop(hbb_chunk)?;

                // Gaze estimation on cropped face
                let ys_gaze = gaze_estimator.forward(&face_crops)?;
                if ys_gaze.is_empty() {
                    println!("No gaze detected in image");
                    continue;
                }

                // Process each result in the batch
                for (result_idx, gaze) in ys_gaze.iter().enumerate() {
                    // Get the corresponding original Hbb
                    let original_hbb_idx = chunk_idx * gaze_estimator.batch() + result_idx;
                    if original_hbb_idx >= hbbs.len() {
                        continue;
                    }
                    let hbb = &hbbs[original_hbb_idx];

                    // Get pitch and yaw from extra
                    let extra = gaze.extra();
                    if let (Some(pitch_tensor), Some(yaw_tensor)) =
                        (extra.get("pitch"), extra.get("yaw"))
                    {
                        let pitch = pitch_tensor.as_slice().unwrap()[0];
                        let yaw = yaw_tensor.as_slice().unwrap()[0];

                        // Calculate gaze projection
                        let (x_center, y_center, x_end, y_end) = usls::calculate_gaze_projection_2d(
                            pitch,
                            yaw,
                            (hbb.xmin(), hbb.ymin(), hbb.xmax(), hbb.ymax()),
                            Some(1.2),
                        );

                        // Draw gaze arrow line
                        usls::draw_arrow_line(
                            &mut rgba,
                            (x_center, y_center),
                            (x_end, y_end),
                            usls::Color::magenta(),
                        );
                    }
                }
            }

            // Annotate and save
            annotator.annotate(&rgba.into(), faces)?.save(format!(
                "{}.jpg",
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/gaze-estimation", gaze_estimator.spec()])?
                    .join(usls::timestamp(None))
                    .display()
            ))?;
        }
    }
    Ok(())
}
