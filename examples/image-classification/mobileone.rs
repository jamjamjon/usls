use anyhow::Result;
use clap::{Parser, ValueEnum};
use usls::{Config, DType, Device};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Kind {
    S0,
    S1,
    S2,
    S3,
    S4_224x224,
    S4_256x256,
    S4_384x384,
    S4_512x512,
}

#[derive(Parser)]
#[command(author, version, about = "Image Classifier Example", long_about = None)]
pub struct MobileOneArgs {
    #[arg(long, default_value = "s0")]
    kind: Kind,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
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

pub fn config(args: &MobileOneArgs) -> Result<Config> {
    let config = match args.kind {
        Kind::S0 => Config::mobileone_s0(),
        Kind::S1 => Config::mobileone_s1(),
        Kind::S2 => Config::mobileone_s2(),
        Kind::S3 => Config::mobileone_s3(),
        Kind::S4_224x224 => Config::mobileone_s4_224x224(),
        Kind::S4_256x256 => Config::mobileone_s4_256x256(),
        Kind::S4_384x384 => Config::mobileone_s4_384x384(),
        Kind::S4_512x512 => Config::mobileone_s4_512x512(),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
