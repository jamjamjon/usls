use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct YoloePromptArgs {
    /// Scale: v8/v11 with s, m, l | 26 with n, s, m, l, x
    #[arg(long, default_value = "s")]
    pub scale: Scale,

    /// Version: 8, 11, 26
    #[arg(long, default_value = "8")]
    pub ver: u8,

    /// Model Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub model_dtype: DType,

    /// Model Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, default_value = "cpu")]
    pub model_device: Device,

    /// Visual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub visual_encoder_dtype: DType,

    /// Visual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, default_value = "cpu")]
    pub visual_encoder_device: Device,

    /// Textual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub textual_encoder_dtype: DType,

    /// Textual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, default_value = "cpu")]
    pub textual_encoder_device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, default_value = "cpu")]
    pub processor_device: Device,

    /// Batch size
    #[arg(long, default_value_t = 1)]
    pub batch: usize,

    /// Min batch size (TensorRT)
    #[arg(long, default_value_t = 1)]
    pub min_batch: usize,

    /// Max batch size (TensorRT)
    #[arg(long, default_value_t = 4)]
    pub max_batch: usize,

    /// Num dry run
    #[arg(long, default_value_t = 3)]
    pub num_dry_run: usize,

    /// Visual prompt image path (required for visual prompts)
    #[arg(long)]
    pub prompt_image: Option<String>,
}

/// Get config based on whether visual prompt is used
pub fn config(args: &YoloePromptArgs) -> Result<Config> {
    if args.prompt_image.is_none() {
        config_textual(args)
    } else {
        config_visual(args)
    }
}

/// Get config for visual prompt model
pub fn config_visual(args: &YoloePromptArgs) -> Result<Config> {
    let config = match (args.ver, &args.scale) {
        (8, Scale::S) => Config::yoloe_v8s_seg_vp(),
        (8, Scale::M) => Config::yoloe_v8m_seg_vp(),
        (8, Scale::L) => Config::yoloe_v8l_seg_vp(),
        (11, Scale::S) => Config::yoloe_11s_seg_vp(),
        (11, Scale::M) => Config::yoloe_11m_seg_vp(),
        (11, Scale::L) => Config::yoloe_11l_seg_vp(),
        (26, Scale::N) => Config::yoloe_26n_seg_vp(),
        (26, Scale::S) => Config::yoloe_26s_seg_vp(),
        (26, Scale::M) => Config::yoloe_26m_seg_vp(),
        (26, Scale::L) => Config::yoloe_26l_seg_vp(),
        (26, Scale::X) => Config::yoloe_26x_seg_vp(),
        _ => anyhow::bail!(
            "Unsupported version {}, scale {} for visual prompt. Try v8/v11 with s, m, l or 26 with n, s, m, l, x.",
            args.ver,
            args.scale,
        ),
    }
    .with_model_dtype(args.model_dtype)
    .with_model_device(args.model_device)
    .with_visual_encoder_dtype(args.visual_encoder_dtype)
    .with_visual_encoder_device(args.visual_encoder_device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}

/// Get config for textual prompt model
pub fn config_textual(args: &YoloePromptArgs) -> Result<Config> {
    let config = match (args.ver, &args.scale) {
        (8, Scale::S) => Config::yoloe_v8s_seg_tp(),
        (8, Scale::M) => Config::yoloe_v8m_seg_tp(),
        (8, Scale::L) => Config::yoloe_v8l_seg_tp(),
        (11, Scale::S) => Config::yoloe_11s_seg_tp(),
        (11, Scale::M) => Config::yoloe_11m_seg_tp(),
        (11, Scale::L) => Config::yoloe_11l_seg_tp(),
        (26, Scale::N) => Config::yoloe_26n_seg_tp(),
        (26, Scale::S) => Config::yoloe_26s_seg_tp(),
        (26, Scale::M) => Config::yoloe_26m_seg_tp(),
        (26, Scale::L) => Config::yoloe_26l_seg_tp(),
        (26, Scale::X) => Config::yoloe_26x_seg_tp(),
        _ => anyhow::bail!(
            "Unsupported version {}, scale {} for textual prompt. Try v8/v11 with s, m, l or 26 with n, s, m, l, x.",
            args.ver,
            args.scale,
        ),
    }
    .with_model_dtype(args.model_dtype)
    .with_model_device(args.model_device)
    .with_textual_encoder_dtype(args.textual_encoder_dtype)
    .with_textual_encoder_device(args.textual_encoder_device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
