use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct Sam3TrackerArgs {
    /// Vision Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "q4f16")]
    pub vision_dtype: DType,

    /// Vision Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub vision_device: Device,

    /// Decoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub decoder_dtype: DType,

    /// Decoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub decoder_device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

    /// Vision encoder batch
    #[arg(long, default_value_t = 1)]
    pub vision_batch: usize,

    /// Decoder batch
    #[arg(long, default_value_t = 1)]
    pub decoder_batch: usize,

    /// num dry run
    #[arg(long, global = true, default_value_t = 0)]
    pub num_dry_run: usize,

    /// Show mask
    #[arg(long, default_value_t = true)]
    pub show_mask: bool,

    /// Prompts: "name;point:x,y,1" or "name;box:x,y,w,h"
    #[arg(short = 'p', long)]
    pub prompt: Vec<String>,
}

pub fn config(args: &Sam3TrackerArgs) -> Result<Config> {
    let config = Config::sam3_tracker()
        .with_visual_encoder_batch_min_opt_max(1, args.vision_batch, 8)
        .with_decoder_batch_min_opt_max(1, args.decoder_batch, 8)
        .with_visual_encoder_dtype(args.vision_dtype)
        .with_visual_encoder_device(args.vision_device)
        .with_decoder_dtype(args.decoder_dtype)
        .with_decoder_device(args.decoder_device)
        .with_num_dry_run_all(args.num_dry_run)
        .with_image_processor_device(args.processor_device);

    Ok(config)
}
