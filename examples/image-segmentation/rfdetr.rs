use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct RfdetrArgs {
    /// Scale: n, s, m, b, l
    #[arg(long, global = true, default_value = "n")]
    pub scale: Scale,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
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

pub fn config(args: &RfdetrArgs) -> Result<Config> {
    let config = match args.scale {
        Scale::N => Config::rfdetr_seg_nano(),
        Scale::S => Config::rfdetr_seg_small(),
        Scale::M => Config::rfdetr_seg_medium(),
        Scale::L => Config::rfdetr_seg_large(),
        Scale::XL => Config::rfdetr_seg_xlarge(),
        Scale::XXL => Config::rfdetr_seg_2xlarge(),
        _ => anyhow::bail!(
            "Unsupported scale: {}. Try n, s, m, l, xl, xxl.",
            args.scale
        ),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
