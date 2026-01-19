use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct YoloePromptFreeArgs {
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

pub fn config(args: &YoloePromptFreeArgs) -> Result<Config> {
    let config = match (args.ver, &args.scale) {
        (8, Scale::S) => Config::yoloe_v8s_seg_pf(),
        (8, Scale::M) => Config::yoloe_v8m_seg_pf(),
        (8, Scale::L) => Config::yoloe_v8l_seg_pf(),
        (11, Scale::S) => Config::yoloe_11s_seg_pf(),
        (11, Scale::M) => Config::yoloe_11m_seg_pf(),
        (11, Scale::L) => Config::yoloe_11l_seg_pf(),
        (26, Scale::N) => Config::yoloe_26n_seg_pf(),
        (26, Scale::S) => Config::yoloe_26s_seg_pf(),
        (26, Scale::M) => Config::yoloe_26m_seg_pf(),
        (26, Scale::L) => Config::yoloe_26l_seg_pf(),
        (26, Scale::X) => Config::yoloe_26x_seg_pf(),
        _ => anyhow::bail!(
            "Unsupported version {} with scale {}. Try v8/11/26 with n, s, m, l, x.",
            args.ver,
            args.scale
        ),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
