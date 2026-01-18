use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct Sam3ImageArgs {
    /// Textual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub textual_encoder_dtype: DType,

    /// Textual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub textual_encoder_device: Device,

    /// Textual encoder batch
    #[arg(long, default_value_t = 2)]
    pub textual_encoder_batch: usize,

    /// Visual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "q4f16")]
    pub visual_encoder_dtype: DType,

    /// Visual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub visual_encoder_device: Device,

    /// Visual encoder batch
    #[arg(long, default_value_t = 1)]
    pub visual_encoder_batch: usize,

    /// Geo-encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "f16")]
    pub geo_encoder_dtype: DType,

    /// Geo-encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub geo_encoder_device: Device,

    /// Geometry encoder batch
    #[arg(long, default_value_t = 4)]
    pub geo_encoder_batch: usize,

    /// Decoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "f16")]
    pub decoder_dtype: DType,

    /// Decoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub decoder_device: Device,

    /// Decoder batch
    #[arg(long, default_value_t = 1)]
    pub decoder_batch: usize,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

    /// num dry run
    #[arg(long, global = true, default_value_t = 3)]
    pub num_dry_run: usize,
}

pub fn config(args: &Sam3ImageArgs) -> Result<Config> {
    let config = Config::sam3_image()
        .with_visual_encoder_batch_min_opt_max(1, args.visual_encoder_batch, 8)
        .with_textual_encoder_batch_min_opt_max(1, args.textual_encoder_batch, 16)
        .with_encoder_batch_min_opt_max(1, args.geo_encoder_batch, 16)
        .with_decoder_batch_min_opt_max(1, args.decoder_batch, 8)
        .with_textual_encoder_device(args.textual_encoder_device)
        .with_textual_encoder_dtype(args.textual_encoder_dtype)
        .with_visual_encoder_device(args.visual_encoder_device)
        .with_visual_encoder_dtype(args.visual_encoder_dtype)
        .with_encoder_device(args.geo_encoder_device)
        .with_encoder_dtype(args.geo_encoder_dtype)
        .with_decoder_device(args.decoder_device)
        .with_decoder_dtype(args.decoder_dtype)
        .with_num_dry_run_all(args.num_dry_run)
        .with_image_processor_device(args.processor_device);

    Ok(config)
}
