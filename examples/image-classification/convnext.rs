use anyhow::Result;
use clap::Parser;
use usls::{Config, DType, Device, Scale};

#[derive(Parser)]
pub struct ConvNextArgs {
    #[arg(long, default_value = "femto")]
    pub scale: Scale,

    #[arg(long, default_value = "2")]
    pub ver: u8,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true)]
    pub processor_device: Option<Device>,

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

pub fn config(args: &ConvNextArgs) -> Result<Config> {
    let mut config = match (args.ver, &args.scale) {
        (1, Scale::T) => Config::convnext_v1_tiny(),
        (1, Scale::S) => Config::convnext_v1_small(),
        (1, Scale::B) => Config::convnext_v1_base(),
        (1, Scale::L) => Config::convnext_v1_large(),
        (2, Scale::A) => Config::convnext_v2_atto(),
        (2, Scale::F) => Config::convnext_v2_femto(),
        (2, Scale::P) => Config::convnext_v2_pico(),
        (2, Scale::N) => Config::convnext_v2_nano(),
        (2, Scale::T) => Config::convnext_v2_tiny(),
        (2, Scale::S) => Config::convnext_v2_small(),
        (2, Scale::B) => Config::convnext_v2_base(),
        (2, Scale::L) => Config::convnext_v2_large(),
        _ => unimplemented!(
            "Unsupported ConvNeXt version {}, scale {}",
            args.ver,
            args.scale
        ),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run);

    if let Some(device) = args.processor_device {
        config = config.with_image_processor_device(device);
    }

    Ok(config)
}
