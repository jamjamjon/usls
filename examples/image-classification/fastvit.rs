use anyhow::Result;
use clap::{Parser, ValueEnum};
use usls::{Config, DType, Device};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Kind {
    T8,
    T8Distill,
    T12,
    T12Distill,
    S12,
    S12Distill,
    Sa12,
    Sa12Distill,
    Sa24,
    Sa24Distill,
    Sa36,
    Sa36Distill,
    Ma36,
    Ma36Distill,
}

#[derive(Parser)]
#[command(author, version, about = "Image Classifier Example", long_about = None)]
pub struct FastViTArgs {
    #[arg(long, default_value = "t8-distill")]
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

pub fn config(args: &FastViTArgs) -> Result<Config> {
    let config = match args.kind {
        Kind::T8 => Config::fastvit_t8(),
        Kind::T8Distill => Config::fastvit_t8_distill(),
        Kind::T12 => Config::fastvit_t12(),
        Kind::T12Distill => Config::fastvit_t12_distill(),
        Kind::S12 => Config::fastvit_s12(),
        Kind::S12Distill => Config::fastvit_s12_distill(),
        Kind::Sa12 => Config::fastvit_sa12(),
        Kind::Sa12Distill => Config::fastvit_sa12_distill(),
        Kind::Sa24 => Config::fastvit_sa24(),
        Kind::Sa24Distill => Config::fastvit_sa24_distill(),
        Kind::Sa36 => Config::fastvit_sa36(),
        Kind::Sa36Distill => Config::fastvit_sa36_distill(),
        Kind::Ma36 => Config::fastvit_ma36(),
        Kind::Ma36Distill => Config::fastvit_ma36_distill(),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
