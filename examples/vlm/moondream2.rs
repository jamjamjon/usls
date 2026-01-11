use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct Moondream2Args {
    /// Scale: 0.5b, 2b
    #[arg(long, default_value = "0.5b")]
    pub scale: Scale,

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

    /// Task: Caption: 0, Vqa: <query>, OpenSetDetection: <query>, etc.
    #[arg(long, default_value = "Caption: 0")]
    pub task: String,
}

pub fn config(args: &Moondream2Args) -> Result<Config> {
    let mut config = match args.scale {
        Scale::Billion(0.5) => Config::moondream2_0_5b(),
        Scale::Billion(2.0) => Config::moondream2_2b(),
        _ => anyhow::bail!("Unsupported Moondream2 scale: {}", args.scale),
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
