use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct BlipArgs {
    /// Variant: v1-base-caption
    #[arg(long, default_value = "v1-base-caption")]
    pub variant: String,

    /// Visual Dtype: fp32
    #[arg(long, default_value = "fp32")]
    pub visual_dtype: DType,

    /// Visual Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub visual_device: Device,

    /// Textual Dtype: fp32
    #[arg(long, default_value = "fp32")]
    pub textual_dtype: DType,

    /// Textual Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub textual_device: Device,

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

    /// Optional prompt for conditional caption
    #[arg(long)]
    pub prompt: Option<String>,
}

pub fn config(args: &BlipArgs) -> Result<Config> {
    let config = match args.variant.as_str() {
        "v1-base-caption" => Config::blip_v1_base_caption(),
        _ => anyhow::bail!("Unsupported BLIP variant: {}", args.variant),
    }
    .with_visual_dtype(args.visual_dtype)
    .with_visual_device(args.visual_device)
    .with_textual_dtype(args.textual_dtype)
    .with_textual_device(args.textual_device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
