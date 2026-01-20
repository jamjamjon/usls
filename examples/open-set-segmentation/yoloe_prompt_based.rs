use anyhow::Result;
use clap::{Args, ValueEnum};
use usls::{Config, DType, Device, Scale};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Kind {
    Textual,
    Visual,
}

#[derive(Args, Debug)]
pub struct YoloePromptArgs {
    /// Scale: s, m, l
    #[arg(long, default_value = "s")]
    pub scale: Scale,

    /// version: 8, 11
    #[arg(long, default_value = "8")]
    pub ver: u8,

    /// Model Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub model_dtype: DType,

    /// Model Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub model_device: Device,

    /// Visual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub visual_encoder_dtype: DType,

    /// Visual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub visual_encoder_device: Device,

    /// Textual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub textual_encoder_dtype: DType,

    /// Textual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub textual_encoder_device: Device,

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

    /// Visual or textual prompt
    #[arg(long, value_enum, default_value = "visual")]
    pub kind: Kind,
}

pub fn config(args: &YoloePromptArgs) -> Result<Config> {
    let mut config = match (args.ver, &args.scale, args.kind) {
        (8, Scale::S, Kind::Visual) => Config::yoloe_v8s_seg_vp(),
        (8, Scale::M, Kind::Visual) => Config::yoloe_v8m_seg_vp(),
        (8, Scale::L, Kind::Visual) => Config::yoloe_v8l_seg_vp(),
        (11, Scale::S, Kind::Visual) => Config::yoloe_11s_seg_vp(),
        (11, Scale::M, Kind::Visual) => Config::yoloe_11m_seg_vp(),
        (11, Scale::L, Kind::Visual) => Config::yoloe_11l_seg_vp(),
        (8, Scale::S, Kind::Textual) => Config::yoloe_v8s_seg_tp(),
        (8, Scale::M, Kind::Textual) => Config::yoloe_v8m_seg_tp(),
        (8, Scale::L, Kind::Textual) => Config::yoloe_v8l_seg_tp(),
        (11, Scale::S, Kind::Textual) => Config::yoloe_11s_seg_tp(),
        (11, Scale::M, Kind::Textual) => Config::yoloe_11m_seg_tp(),
        (11, Scale::L, Kind::Textual) => Config::yoloe_11l_seg_tp(),
        (26, Scale::N, Kind::Visual) => Config::yoloe_26n_seg_vp(),
        (26, Scale::S, Kind::Visual) => Config::yoloe_26s_seg_vp(),
        (26, Scale::M, Kind::Visual) => Config::yoloe_26m_seg_vp(),
        (26, Scale::L, Kind::Visual) => Config::yoloe_26l_seg_vp(),
        (26, Scale::X, Kind::Visual) => Config::yoloe_26x_seg_vp(),
        (26, Scale::N, Kind::Textual) => Config::yoloe_26n_seg_tp(),
        (26, Scale::S, Kind::Textual) => Config::yoloe_26s_seg_tp(),
        (26, Scale::M, Kind::Textual) => Config::yoloe_26m_seg_tp(),
        (26, Scale::L, Kind::Textual) => Config::yoloe_26l_seg_tp(),
        (26, Scale::X, Kind::Textual) => Config::yoloe_26x_seg_tp(),
        _ => anyhow::bail!(
            "Unsupported version {}, scale {}, kind {:?}. Try v8/v11 with s, m, l, 26 with n, s, m, l, x and visual/textual.",
            args.ver,
            args.scale,
            args.kind
        ),
    }
    .with_model_dtype(args.model_dtype)
    .with_model_device(args.model_device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    // Conditionally set encoder dtype/device based on kind
    match args.kind {
        Kind::Textual => {
            config = config
                .with_textual_encoder_dtype(args.textual_encoder_dtype)
                .with_textual_encoder_device(args.textual_encoder_device);
        }
        Kind::Visual => {
            config = config
                .with_visual_encoder_dtype(args.visual_encoder_dtype)
                .with_visual_encoder_device(args.visual_encoder_device);
        }
    }

    Ok(config)
}
