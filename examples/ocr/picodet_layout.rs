use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct PicodetLayoutArgs {
    /// Variant: 1x, l-3cls, l-17cls
    #[arg(long, default_value = "1x")]
    pub variant: String,

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

pub fn config(args: &PicodetLayoutArgs) -> Result<Config> {
    let config = match args.variant.as_str() {
        "1x" => Config::picodet_layout_1x(),
        "l-3cls" => Config::picodet_l_layout_3cls(),
        "l-17cls" => Config::picodet_l_layout_17cls(),
        _ => anyhow::bail!("Unsupported PicoDet Layout variant: {}", args.variant),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
