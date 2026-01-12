use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct DinoArgs {
    /// Variant: v2-s, v2-b, v3-s, v3-s-plus, v3-b, v3-l, v3-l-sat493m, v3-h-plus
    #[arg(long, default_value = "v2-s")]
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

pub fn config(args: &DinoArgs) -> Result<Config> {
    let config = match args.variant.as_str() {
        "v2-s" => Config::dinov2_small(),
        "v2-b" => Config::dinov2_base(),
        "v3-s" => Config::dinov3_vits16_lvd1689m(),
        "v3-s-plus" => Config::dinov3_vits16plus_lvd1689m(),
        "v3-b" => Config::dinov3_vitb16_lvd1689m(),
        "v3-l" => Config::dinov3_vitl16_lvd1689m(),
        "v3-l-sat493m" => Config::dinov3_vitl16_sat493m(),
        "v3-h-plus" => Config::dinov3_vith16plus_lvd1689m(),
        _ => anyhow::bail!("Unsupported DINO variant: {}", args.variant),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
