use anyhow::Result;
use clap::{Parser, ValueEnum};
use usls::{Config, DType, Scale};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Kind {
    Mono,
    Multi,
    Metric,
}

#[derive(Parser, Debug)]
pub struct DepthAnythingArgs {
    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: String,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: String,

    /// Batch size
    #[arg(long, global = true, default_value_t = 1)]
    pub batch: usize,

    /// Scale: s, b, l
    #[arg(long, default_value = "s")]
    pub scale: Scale,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub dtype: DType,

    /// version: 1, 2, 3
    #[arg(long, default_value = "2", value_parser = clap::value_parser!(u8).range(1..=3))]
    pub ver: u8,

    /// Kind: mono, multi
    #[arg(long, default_value = "mono")]
    pub kind: Kind,
}

pub fn config(args: &DepthAnythingArgs) -> Result<Config> {
    let config = match (args.ver, &args.scale, args.kind) {
        (1, Scale::S, _) => Config::depth_anything_v1_small(),
        (1, Scale::B, _) => Config::depth_anything_v1_base(),
        (1, Scale::L, _) => Config::depth_anything_v1_large(),
        (2, Scale::S, _) => Config::depth_anything_v2_small(),
        (2, Scale::B, _) => Config::depth_anything_v2_base(),
        (2, Scale::L, _) => Config::depth_anything_v2_large(),
        (3, Scale::L, Kind::Mono) => Config::depth_anything_v3_mono_large(),
        (3, Scale::L, Kind::Metric) => Config::depth_anything_v3_metric_large(),
        (3, Scale::S, Kind::Multi) => Config::depth_anything_v3_small(),
        (3, Scale::B, Kind::Multi) => Config::depth_anything_v3_base(),
        (3, Scale::L, Kind::Multi) => Config::depth_anything_v3_large(),
        _ => unimplemented!(
            "Unsupported configuration: version: {} scale: {} kind: {:?}",
            args.ver,
            args.scale,
            args.kind
        ),
    }
    .with_batch_size_all_min_opt_max(1, args.batch, 4)
    .with_device_all(args.device.parse()?)
    .with_image_processor_device(args.processor_device.parse()?)
    .with_model_dtype(args.dtype);

    Ok(config)
}
