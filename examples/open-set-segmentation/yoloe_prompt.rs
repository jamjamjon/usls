use anyhow::Result;
use clap::{Args, ValueEnum};
use usls::{Config, DType, Scale};

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

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: String,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: String,

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
    let config = match (args.ver, &args.scale, args.kind) {
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
        _ => anyhow::bail!(
            "Unsupported version {}, scale {}, kind {:?}. Try v8/v11 with s, m, l. and visual/textual.",
            args.ver,
            args.scale,
            args.kind
        ),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device.parse()?)
    // .with_textual_encoder_dtype("fp16".parse()?)  // optional
    .with_image_processor_device(args.processor_device.parse()?)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run);

    Ok(config)
}
