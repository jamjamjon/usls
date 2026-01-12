use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, SamKind, Scale};

#[derive(Args, Debug)]
pub struct SamArgs {
    /// SAM kind: sam, sam2, mobilesam, samhq, edgesam
    #[arg(long, default_value = "samhq")]
    pub kind: SamKind,

    /// Scale: t, s, b, l (for sam2 only)
    #[arg(long, default_value = "t")]
    pub scale: Scale,

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

pub fn config(args: &SamArgs) -> Result<Config> {
    let config = match args.kind {
        SamKind::Sam => Config::sam_v1_base(),
        SamKind::Sam2 => match args.scale {
            Scale::T => Config::sam2_tiny(),
            Scale::S => Config::sam2_small(),
            Scale::B => Config::sam2_base_plus(),
            _ => anyhow::bail!("Unsupported SAM2 scale: {}. Try t, s, b.", args.scale),
        },
        SamKind::MobileSam => Config::mobile_sam_tiny(),
        SamKind::SamHq => Config::sam_hq_tiny(),
        SamKind::EdgeSam => Config::edge_sam_3x(),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
