use anyhow::Result;
use clap::Parser;
use usls::{Config, DType, Device, Version};

#[derive(Parser)]
pub struct RmbgArgs {
    /// Version: 1.4, 2.0
    #[arg(long, default_value = "1.4")]
    pub ver: Version,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
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

pub fn config(args: &RmbgArgs) -> Result<Config> {
    let mut config = match args.ver {
        Version(1, 4, _) => Config::rmbg1_4(),
        Version(2, 0, _) => Config::rmbg2_0(),
        _ => unimplemented!(),
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
