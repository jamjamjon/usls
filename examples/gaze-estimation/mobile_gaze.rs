use anyhow::Result;
use clap::{Args, ValueEnum};
use usls::{Config, DType, Device};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Variant {
    #[clap(name = "r18")]
    R18,
    #[clap(name = "r34")]
    R34,
    #[clap(name = "r50")]
    R50,
    #[clap(name = "mobilenet-v2")]
    MobileNetV2,
    #[clap(name = "mobileone-s0")]
    MobileOneS0,
}

#[derive(Args, Debug)]
pub struct MobileGazeArgs {
    /// Variant: r18, r34, r50, mobilenet-v2, mobileone-s0
    #[arg(long, global = true, default_value = "mobileone-s0")]
    pub variant: Variant,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, global = true, default_value = "f16")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

    /// Batch size
    #[arg(long, global = true, default_value_t = 1)]
    pub batch: usize,

    /// Min batch size (TensorRT)
    #[arg(long, global = true, default_value_t = 1)]
    pub min_batch: usize,

    /// Max batch size (TensorRT)
    #[arg(long, global = true, default_value_t = 4)]
    pub max_batch: usize,

    /// num dry run
    #[arg(long, global = true, default_value_t = 3)]
    pub num_dry_run: usize,
}

pub fn config(args: &MobileGazeArgs) -> Result<Config> {
    let config = match args.variant {
        Variant::R18 => Config::mobile_gaze_r18(),
        Variant::R34 => Config::mobile_gaze_r34(),
        Variant::R50 => Config::mobile_gaze_r50(),
        Variant::MobileNetV2 => Config::mobile_gaze_mobilenet_v2(),
        Variant::MobileOneS0 => Config::mobile_gaze_mobileone_s0(),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
