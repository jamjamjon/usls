use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct HrnetArgs {
    /// Backbone width: w32 or w48
    #[arg(long, default_value = "w48")]
    pub width: String,

    /// Use COCO 17 keypoints (true = body) or COCO-WholeBody 133 keypoints (false)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub is_coco: bool,

    /// Use 384x288 input (true) or 256x192 (false). WholeBody always uses 384x288.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub hires: bool,

    /// Optional local model file path (overrides the built-in selection)
    #[arg(long)]
    pub model: Option<String>,

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

pub fn config(args: &HrnetArgs) -> Result<Config> {
    let mut config = match (args.width.as_str(), args.is_coco) {
        ("w32", true) => {
            if args.hires {
                Config::hrnet_w32_17_384()
            } else {
                Config::hrnet_w32_17()
            }
        }
        ("w48", true) => {
            if args.hires {
                Config::hrnet_w48_17_384()
            } else {
                Config::hrnet_w48_17()
            }
        }
        ("w32", false) => Config::hrnet_w32_133(),
        ("w48", false) => Config::hrnet_w48_133(),
        (w, _) => anyhow::bail!("Unsupported HRNet width: {w} (expected w32 or w48)"),
    };

    // Allow overriding with a local file (e.g. the sample end2end.hrnet_w48.onnx)
    if let Some(model) = &args.model {
        config = config.with_model_file(model);
    }

    let config = config
        .with_model_dtype(args.dtype)
        .with_model_device(args.device)
        .with_model_batch_size_min_opt_max(args.min_batch, args.batch, args.max_batch)
        .with_model_num_dry_run(args.num_dry_run)
        .with_image_processor_device(args.processor_device);

    Ok(config)
}
