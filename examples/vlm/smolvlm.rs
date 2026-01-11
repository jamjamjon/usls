use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale, Version};

#[derive(Args, Debug)]
pub struct SmolvlmArgs {
    /// Scale: 256m, 500m
    #[arg(long, default_value = "256m")]
    pub scale: Scale,

    /// Version: 1, 2
    #[arg(long, default_value = "2")]
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

    /// Prompt
    #[arg(long, default_value = "Can you describe this image?")]
    pub prompt: String,

    /// Max tokens
    #[arg(long, default_value_t = 512)]
    pub max_tokens: u64,

    /// Ignore end-of-sequence token
    #[arg(long, default_value_t = false)]
    pub ignore_eos: bool,
}

pub fn config(args: &SmolvlmArgs) -> Result<Config> {
    let config = match (args.scale.clone(), args.ver) {
        (Scale::Million(256.0), Version(1, 0, _)) => Config::smolvlm_256m(),
        (Scale::Million(500.0), Version(1, 0, _)) => Config::smolvlm_500m(),
        (Scale::Million(256.0), Version(2, 0, _)) => Config::smolvlm2_256m(),
        (Scale::Million(500.0), Version(2, 0, _)) => Config::smolvlm2_500m(),
        _ => anyhow::bail!(
            "Unsupported SmolVLM scale/version: {} v{}",
            args.scale,
            args.ver
        ),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_image_processor_device(args.processor_device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_max_tokens(args.max_tokens)
    .with_ignore_eos(args.ignore_eos);

    Ok(config)
}
