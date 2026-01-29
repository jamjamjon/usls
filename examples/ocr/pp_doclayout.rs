use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Version};

#[derive(Args, Debug)]
pub struct PPDoclayoutArgs {
    /// Model version
    #[arg(short, long, default_value = "2")]
    pub ver: Version,

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
}

pub fn config(args: &PPDoclayoutArgs) -> Result<Config> {
    let config = match args.ver {
        Version(1, _, _) => Config::pp_doclayout_v1_plus_l(),
        Version(2, _, _) => Config::pp_doclayout_v2(),
        _ => anyhow::bail!("Unsupported version"),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
