use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct RtmposeArgs {
    /// Scale: t, s, m, l, x
    #[arg(long, default_value = "t")]
    pub scale: Scale,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp32")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true)]
    pub processor_device: Option<Device>,

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

    /// Use COCO 17 keypoints (false = Halpe 26)
    #[arg(long, default_value_t = true)]
    pub is_coco: bool,
}

pub fn config(args: &RtmposeArgs) -> Result<Config> {
    let mut config = match args.scale {
        Scale::T => match args.is_coco {
            true => Config::rtmpose_17_t(),
            false => Config::rtmpose_26_t(),
        },
        Scale::S => match args.is_coco {
            true => Config::rtmpose_17_s(),
            false => Config::rtmpose_26_s(),
        },
        Scale::M => match args.is_coco {
            true => Config::rtmpose_17_m(),
            false => Config::rtmpose_26_m(),
        },
        Scale::L => match args.is_coco {
            true => Config::rtmpose_17_l(),
            false => Config::rtmpose_26_l(),
        },
        Scale::X => match args.is_coco {
            true => Config::rtmpose_17_x(),
            false => Config::rtmpose_26_x(),
        },
        _ => anyhow::bail!("Unsupported RTMPose scale: {}", args.scale),
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run);

    if let Some(device) = args.processor_device {
        config = config.with_image_processor_device(device);
    }

    Ok(config)
}
