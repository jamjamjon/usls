use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct DbArgs {
    /// variant: ppocr-v3-ch, ppocr-v4-ch, ppocr-v4-server-ch, ppocr-v5-mobile, ppocr-v5-server, mobilenet-v3-large, resnet34, resnet50
    #[arg(long, default_value = "ppocr-v4-ch")]
    pub variant: String,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
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

    /// Show HBBs
    #[arg(long)]
    pub show_hbbs: bool,

    /// Show OBBs
    #[arg(long)]
    pub show_obbs: bool,

    /// Show HBBs confidence
    #[arg(long)]
    pub show_hbbs_conf: bool,

    /// Show OBBs confidence
    #[arg(long)]
    pub show_obbs_conf: bool,

    /// Show polygons confidence
    #[arg(long)]
    pub show_polygons_conf: bool,
}

pub fn config(args: &DbArgs) -> Result<Config> {
    let config = match args.variant.as_str() {
        "ppocr-v3-ch" => Config::ppocr_det_v3_ch(),
        "ppocr-v4-ch" => Config::ppocr_det_v4_ch(),
        "ppocr-v4-server-ch" => Config::ppocr_det_v4_server_ch(),
        "ppocr-v5-mobile" => Config::ppocr_det_v5_mobile(),
        "ppocr-v5-server" => Config::ppocr_det_v5_server(),
        "mobilenet-v3-large" => Config::db_mobilenet_v3_large(),
        "resnet34" => Config::db_resnet34(),
        "resnet50" => Config::db_resnet50(),
        _ => anyhow::bail!("Unsupported DB variant: {}", args.variant),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
