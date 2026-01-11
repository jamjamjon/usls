use anyhow::Result;
use clap::Parser;
use usls::{Config, DType, Device};

#[derive(Parser, Debug)]
pub struct DepthProArgs {
    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true)]
    pub processor_device: Option<Device>,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "q4f16")]
    pub dtype: DType,
}

pub fn config(args: &DepthProArgs) -> Result<Config> {
    let mut config = Config::depth_pro()
        .with_dtype_all(args.dtype)
        .with_device_all(args.device)
        .with_batch_size_all_min_opt_max(1, 1, 1)
        .with_num_dry_run_all(0);

    if let Some(device) = args.processor_device {
        config = config.with_image_processor_device(device);
    }

    Ok(config)
}
