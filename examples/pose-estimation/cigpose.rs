use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct CigposeArgs {
    /// Scale: l, x
    #[arg(long, default_value = "l")]
    pub scale: Scale,

    /// Use COCO 17 keypoints (true = body) or COCO-WholeBody 133 keypoints (false)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub is_coco: bool,

    /// Use 384x288 input (true) or 256x192 (false). X scale always uses 384x288.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub hires: bool,

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
}

pub fn config(args: &CigposeArgs) -> Result<Config> {
    let config = match (args.scale.clone(), args.is_coco) {
        (Scale::L, true) => {
            if args.hires {
                Config::cigpose_17_l_384()
            } else {
                Config::cigpose_17_l()
            }
        }
        (Scale::L, false) => {
            if args.hires {
                Config::cigpose_133_l_384()
            } else {
                Config::cigpose_133_l()
            }
        }
        (Scale::X, false) => Config::cigpose_133_x_384(),
        (Scale::X, true) => anyhow::bail!("CIGPose x scale is only available for COCO-WholeBody"),
        (scale, _) => anyhow::bail!("Unsupported CIGPose scale: {scale} (expected l or x)"),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_model_batch_size_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_model_num_dry_run(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
