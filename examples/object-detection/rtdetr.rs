use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Scale, Version};

#[derive(Args, Debug)]
pub struct RtdetrArgs {
    /// Scale: r18, r18-obj365, r34, r50, r50-obj365, r101, r101-obj365 (v1); s, m, ms, l, x (v2/v4)
    #[arg(long, default_value = "r18-obj365")]
    pub scale: Scale,

    /// Version: 1, 2, 4
    #[arg(long, default_value = "1")]
    pub ver: Version,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "f16")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: String,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: String,

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

pub fn config(args: &RtdetrArgs) -> Result<Config> {
    let config = match args.ver {
        Version(1, 0, _) => match args.scale {
            Scale::Named(ref name) if name == "r18" => Config::rtdetr_v1_r18(),
            Scale::Named(ref name) if name == "r18-obj365" => Config::rtdetr_v1_r18_obj365(),
            Scale::Named(ref name) if name == "r34" => Config::rtdetr_v1_r34(),
            Scale::Named(ref name) if name == "r50" => Config::rtdetr_v1_r50(),
            Scale::Named(ref name) if name == "r50-obj365" => Config::rtdetr_v1_r50_obj365(),
            Scale::Named(ref name) if name == "r101" => Config::rtdetr_v1_r101(),
            Scale::Named(ref name) if name == "r101-obj365" => Config::rtdetr_v1_r101_obj365(),
            _ => anyhow::bail!("Unsupported scale for RT-DETRv1: {}", args.scale),
        },
        Version(2, 0, _) => match args.scale {
            Scale::S => Config::rtdetr_v2_s(),
            Scale::M => Config::rtdetr_v2_m(),
            Scale::Named(ref name) if name == "ms" => Config::rtdetr_v2_ms(),
            Scale::L => Config::rtdetr_v2_l(),
            Scale::X => Config::rtdetr_v2_x(),
            _ => anyhow::bail!("Unsupported scale for RT-DETRv2: {}", args.scale),
        },
        Version(4, 0, _) => match args.scale {
            Scale::S => Config::rtdetr_v4_s(),
            Scale::M => Config::rtdetr_v4_m(),
            Scale::L => Config::rtdetr_v4_l(),
            Scale::X => Config::rtdetr_v4_x(),
            _ => anyhow::bail!("Unsupported scale for RT-DETRv4: {}", args.scale),
        },
        _ => anyhow::bail!("Unsupported version: {}", args.ver),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device.parse()?)
    .with_image_processor_device(args.processor_device.parse()?)
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run);

    Ok(config)
}
