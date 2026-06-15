use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct EcdetsegArgs {
    /// Scale: s, m, l, x
    #[arg(long, global = true, default_value = "m")]
    pub scale: Scale,

    /// Optional local ONNX model path (overrides the released asset)
    #[arg(long)]
    pub model: Option<String>,

    /// Dtype: fp32, fp16
    #[arg(long, default_value = "fp16")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, tensorrt:0, trt-rtx:0, coreml, etc.
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

pub fn config(args: &EcdetsegArgs) -> Result<Config> {
    let config = match &args.model {
        Some(path) => Config::ecdetseg().with_model_file(path),
        None => match args.scale {
            Scale::S => Config::ecdetseg_s(),
            Scale::M => Config::ecdetseg_m(),
            Scale::L => Config::ecdetseg_l(),
            Scale::X => Config::ecdetseg_x(),
            _ => anyhow::bail!("Unsupported scale: {}. Try s, m, l, x.", args.scale),
        },
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
