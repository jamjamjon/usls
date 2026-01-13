use anyhow::Result;
use clap::{Parser, ValueEnum};
use usls::{Config, DType, Device};

/// Model variant
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Kind {
    Base,
    BaseEnsemble,
    BaseFt,
}

#[derive(Parser)]
#[command(author, version, about = "OWLv2 Example", long_about = None)]
pub struct Owlv2Args {
    /// Model variant
    #[arg(long, value_enum, default_value = "base")]
    kind: Kind,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

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

pub fn config(args: &Owlv2Args) -> Result<Config> {
    let config = match args.kind {
        Kind::Base => Config::owlv2_base(),
        Kind::BaseEnsemble => Config::owlv2_base_ensemble(),
        Kind::BaseFt => Config::owlv2_base_ft(),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_class_confs(&[0.1])
    .with_model_num_dry_run(args.num_dry_run)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
