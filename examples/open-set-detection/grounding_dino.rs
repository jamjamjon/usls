use anyhow::Result;
use clap::{Parser, ValueEnum};
use usls::{Config, DType, Device};

/// Model variant
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Kind {
    Tiny,
    Base,
    LlmdetTiny,
    LlmdetBase,
    LlmdetLarge,
    MMGDINOTinyO365v1GoldgGritV3det,
    MMGDINOTinyO365v1GoldgGrit,
    MMGDINOTinyO365v1GoldgV3det,
    MMGDINOTinyO365v1Goldg,
    MMGDINOBasO365v1GoldgV3det,
    MMGDINOBasAll,
    MMGDINOLargeO365v2Oiv6Goldg,
    MMGDINOLargeAll,
}

#[derive(Parser)]
#[command(author, version, about = "Grounding DINO Example", long_about = None)]
pub struct GroundingDINOArgs {
    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "q8")]
    pub dtype: DType,

    /// Model variant
    #[arg(long, value_enum, default_value = "llmdet-tiny")]
    kind: Kind,

    /// Token level class
    #[arg(long)]
    token_level_class: bool,

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

pub fn config(args: &GroundingDINOArgs) -> Result<Config> {
    let mut config = match args.kind {
        Kind::Tiny => Config::grounding_dino_tiny(),
        Kind::Base => Config::grounding_dino_base(),
        Kind::LlmdetTiny => Config::llmdet_tiny(),
        Kind::LlmdetBase => Config::llmdet_base(),
        Kind::LlmdetLarge => Config::llmdet_large(),
        Kind::MMGDINOTinyO365v1GoldgGritV3det => Config::mm_gdino_tiny_o365v1_goldg_grit_v3det(),
        Kind::MMGDINOTinyO365v1GoldgGrit => Config::mm_gdino_tiny_o365v1_goldg_grit(),
        Kind::MMGDINOTinyO365v1Goldg => Config::mm_gdino_tiny_o365v1_goldg(),
        Kind::MMGDINOTinyO365v1GoldgV3det => Config::mm_gdino_tiny_o365v1_goldg_v3det(),
        Kind::MMGDINOBasO365v1GoldgV3det => Config::mm_gdino_base_o365v1_goldg_v3det(),
        Kind::MMGDINOBasAll => Config::mm_gdino_base_all(),
        Kind::MMGDINOLargeO365v2Oiv6Goldg => Config::mm_gdino_large_o365v2_oiv6_goldg(),
        Kind::MMGDINOLargeAll => Config::mm_gdino_large_all(),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_class_confs(&[0.35])
    .with_text_confs(&[0.25])
    .with_model_num_dry_run(args.num_dry_run)
    .with_token_level_class(args.token_level_class)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch);

    if let Some(device) = args.processor_device {
        config = config.with_image_processor_device(device);
    }

    Ok(config)
}
