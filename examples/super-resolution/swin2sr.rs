use anyhow::Result;
use clap::Parser;
use usls::{Config, DType, Device};

#[derive(Parser)]
#[command(author, version, about = "Swin2SR Example", long_about = None)]
pub struct Swin2SRArgs {
    /// Kind: lightweight, classical-2x, classical-4x, compressed, realworld
    #[arg(long, default_value = "lightweight")]
    kind: String,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true)]
    pub processor_device: Option<Device>,

    /// num dry run
    #[arg(long, global = true, default_value_t = 3)]
    pub num_dry_run: usize,

    /// Batch size
    #[arg(long, global = true, default_value_t = 1)]
    pub batch: usize,

    /// Min batch size (TensorRT)
    #[arg(long, global = true, default_value_t = 1)]
    pub min_batch: usize,

    /// Max batch size (TensorRT)
    #[arg(long, global = true, default_value_t = 4)]
    pub max_batch: usize,
}

pub fn config(args: &Swin2SRArgs) -> Result<Config> {
    let mut config = match args.kind.as_str() {
        "lightweight" => Config::swin2sr_lightweight_x2_64(),
        "classical-2x" => Config::swin2sr_classical_x2_64(),
        "classical-4x" => Config::swin2sr_classical_x4_64(),
        "compressed" => Config::swin2sr_compressed_x4_48(),
        "realworld" => Config::swin2sr_realworld_x4_64_bsrgan_psnr(),
        _ => unreachable!(),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run);

    if let Some(device) = args.processor_device {
        config = config.with_image_processor_device(device);
    }

    Ok(config)
}
