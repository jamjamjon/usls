use anyhow::Result;
use clap::{Parser, ValueEnum};
use usls::{Config, DType};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Kind {
    Rrdb2x,
    Grl4x,
}

#[derive(Parser)]
#[command(author, version, about = "APISR Example", long_about = None)]
pub struct APISRArgs {
    #[arg(long, value_enum, default_value_t = Kind::Rrdb2x)]
    kind: Kind,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: String,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: String,

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

pub fn config(args: &APISRArgs) -> Result<Config> {
    let config = match args.kind {
        Kind::Grl4x => Config::apisr_grl_4x(),
        Kind::Rrdb2x => Config::apisr_rrdb_2x(),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device.parse()?)
    .with_image_processor_device(args.processor_device.parse()?)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run);

    Ok(config)
}
