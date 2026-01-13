use anyhow::Result;
use clap::Args;
use usls::{vlm::TrOCRKind, Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct TrocrArgs {
    /// Scale: s, b
    #[arg(long, default_value = "s")]
    pub scale: Scale,

    /// Kind: printed, handwritten
    #[arg(long, default_value = "printed")]
    pub kind: TrOCRKind,

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

pub fn config(args: &TrocrArgs) -> Result<Config> {
    let config = match args.scale {
        Scale::S => match args.kind {
            TrOCRKind::Printed => Config::trocr_small_printed(),
            TrOCRKind::HandWritten => Config::trocr_small_handwritten(),
        },
        Scale::B => match args.kind {
            TrOCRKind::Printed => Config::trocr_base_printed(),
            TrOCRKind::HandWritten => Config::trocr_base_handwritten(),
        },
        _ => anyhow::bail!("Unsupported TrOCR scale: {}", args.scale),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
