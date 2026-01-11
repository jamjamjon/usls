use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct FastArgs {
    /// Scale: t, s, b
    #[arg(long, default_value = "t")]
    pub scale: Scale,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
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

    /// Show polygons confidence
    #[arg(long)]
    pub show_polygons_conf: bool,

    /// Show polygons id
    #[arg(long)]
    pub show_polygons_id: bool,

    /// Show polygons name
    #[arg(long)]
    pub show_polygons_name: bool,
}

pub fn config(args: &FastArgs) -> Result<Config> {
    let config = match args.scale {
        Scale::T => Config::fast_tiny(),
        Scale::S => Config::fast_small(),
        Scale::B => Config::fast_base(),
        _ => anyhow::bail!("Unsupported FAST scale: {}", args.scale),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_image_processor_device(args.processor_device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run);

    Ok(config)
}
