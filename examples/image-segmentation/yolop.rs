use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct YolopArgs {
    /// Variant: v2-480x800, v2-736x1280
    #[arg(long, default_value = "v2-480x800")]
    pub variant: String,

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

pub fn config(args: &YolopArgs) -> Result<Config> {
    let mut config = match args.variant.as_str() {
        "v2-480x800" => Config::yolop_v2_480x800(),
        "v2-736x1280" => Config::yolop_v2_736x1280(),
        _ => anyhow::bail!("Unsupported YOLOPv2 variant: {}", args.variant),
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
