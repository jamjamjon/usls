use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct Moondream2Args {
    /// Scale: 0.5b, 2b
    #[arg(long, default_value = "0.5b")]
    pub scale: Scale,

    /// Visual Encoder Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub visual_encoder_dtype: DType,

    /// Visual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub visual_encoder_device: Device,

    /// Visual Projection Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub visual_projection_dtype: DType,

    /// Visual Projection Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub visual_projection_device: Device,

    /// Textual Encoder Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub textual_encoder_dtype: DType,

    /// Textual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub textual_encoder_device: Device,

    /// Textual Decoder Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub textual_decoder_dtype: DType,

    /// Textual Decoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub textual_decoder_device: Device,

    /// Coord Encoder Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub coord_encoder_dtype: DType,

    /// Coord Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub coord_encoder_device: Device,

    /// Coord Decoder Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub coord_decoder_dtype: DType,

    /// Coord Decoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub coord_decoder_device: Device,

    /// Size Encoder Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub size_encoder_dtype: DType,

    /// Size Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub size_encoder_device: Device,

    /// Size Decoder Dtype: int4 int8
    #[arg(long, default_value = "int4")]
    pub size_decoder_dtype: DType,

    /// Size Decoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub size_decoder_device: Device,

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

    /// Task: Caption: 0, Vqa: <query>, OpenSetDetection: <query>, etc.
    #[arg(long, default_value = "Caption: 0")]
    pub task: String,
}

pub fn config(args: &Moondream2Args) -> Result<Config> {
    let config = match args.scale {
        Scale::Billion(0.5) => Config::moondream2_0_5b(),
        Scale::Billion(2.0) => Config::moondream2_2b(),
        _ => anyhow::bail!("Unsupported Moondream2 scale: {}", args.scale),
    }
    .with_visual_encoder_dtype(args.visual_encoder_dtype)
    .with_visual_encoder_device(args.visual_encoder_device)
    .with_visual_projection_dtype(args.visual_projection_dtype)
    .with_visual_projection_device(args.visual_projection_device)
    .with_textual_encoder_dtype(args.textual_encoder_dtype)
    .with_textual_encoder_device(args.textual_encoder_device)
    .with_textual_decoder_dtype(args.textual_decoder_dtype)
    .with_textual_decoder_device(args.textual_decoder_device)
    .with_coord_encoder_dtype(args.coord_encoder_dtype)
    .with_coord_encoder_device(args.coord_encoder_device)
    .with_coord_decoder_dtype(args.coord_decoder_dtype)
    .with_coord_decoder_device(args.coord_decoder_device)
    .with_size_encoder_dtype(args.size_encoder_dtype)
    .with_size_encoder_device(args.size_encoder_device)
    .with_size_decoder_dtype(args.size_decoder_dtype)
    .with_size_decoder_device(args.size_decoder_device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
