use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct ClipArgs {
    /// Variant: clip-b16, clip-b32, clip-l14, jina-clip-v1, jina-clip-v2, mobileclip-s0, mobileclip-s1, mobileclip-s2, mobileclip-b, mobileclip-blt, mobileclip2-s0, mobileclip2-s2, mobileclip2-s4, mobileclip2-b, mobileclip2-l14
    #[arg(long, default_value = "mobileclip2-s0")]
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

pub fn config(args: &ClipArgs) -> Result<Config> {
    let config = match args.variant.as_str() {
        "clip-b16" => Config::clip_vit_b16(),
        "clip-b32" => Config::clip_vit_b32(),
        "clip-l14" => Config::clip_vit_l14(),
        "jina-clip-v1" => Config::jina_clip_v1(),
        "jina-clip-v2" => Config::jina_clip_v2(),
        "mobileclip-s0" => Config::mobileclip_s0(),
        "mobileclip-s1" => Config::mobileclip_s1(),
        "mobileclip-s2" => Config::mobileclip_s2(),
        "mobileclip-b" => Config::mobileclip_b(),
        "mobileclip-blt" => Config::mobileclip_blt(),
        "mobileclip2-s0" => Config::mobileclip2_s0(),
        "mobileclip2-s2" => Config::mobileclip2_s2(),
        "mobileclip2-s4" => Config::mobileclip2_s4(),
        "mobileclip2-b" => Config::mobileclip2_b(),
        "mobileclip2-l14" => Config::mobileclip2_l14(),
        _ => anyhow::bail!("Unsupported CLIP variant: {}", args.variant),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
