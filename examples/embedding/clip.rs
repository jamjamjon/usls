use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct ClipArgs {
    /// Variant: clip-b16, clip-b32, clip-l14, jina-clip-v1, jina-clip-v2, mobileclip-s0, mobileclip-s1, mobileclip-s2, mobileclip-b, mobileclip-blt, mobileclip2-s0, mobileclip2-s2, mobileclip2-s4, mobileclip2-b, mobileclip2-l14, siglip-b16-224, siglip-b16-256, siglip-b16-384, siglip-b16-512, siglip-l16-256, siglip-l16-384, siglip2-b16-224, siglip2-b16-256, siglip2-b16-384, siglip2-b16-512, siglip2-l16-256, siglip2-l16-384, siglip2-l16-512, siglip2-so400m-patch14-224, siglip2-so400m-patch14-384, siglip2-so400m-patch16-256, siglip2-so400m-patch16-384, siglip2-so400m-patch16-512
    #[arg(long, default_value = "mobileclip2-s0")]
    pub variant: String,

    /// Visual Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub visual_dtype: DType,

    /// Visual Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub visual_device: Device,

    /// Textual Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
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
        "siglip-b16-224" => Config::siglip_b16_224(),
        "siglip-b16-256" => Config::siglip_b16_256(),
        "siglip-b16-384" => Config::siglip_b16_384(),
        "siglip-b16-512" => Config::siglip_b16_512(),
        "siglip-l16-256" => Config::siglip_l16_256(),
        "siglip-l16-384" => Config::siglip_l16_384(),
        "siglip2-b16-224" => Config::siglip2_b16_224(),
        "siglip2-b16-256" => Config::siglip2_b16_256(),
        "siglip2-b16-384" => Config::siglip2_b16_384(),
        "siglip2-b16-512" => Config::siglip2_b16_512(),
        "siglip2-l16-256" => Config::siglip2_l16_256(),
        "siglip2-l16-384" => Config::siglip2_l16_384(),
        "siglip2-l16-512" => Config::siglip2_l16_512(),
        "siglip2-so400m-patch14-224" => Config::siglip2_so400m_patch14_224(),
        "siglip2-so400m-patch14-384" => Config::siglip2_so400m_patch14_384(),
        "siglip2-so400m-patch16-256" => Config::siglip2_so400m_patch16_256(),
        "siglip2-so400m-patch16-384" => Config::siglip2_so400m_patch16_384(),
        "siglip2-so400m-patch16-512" => Config::siglip2_so400m_patch16_512(),
        _ => anyhow::bail!("Unsupported CLIP variant: {}", args.variant),
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
