use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device};

#[derive(Args, Debug)]
pub struct BiRefNetArgs {
    /// Model file
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Variant: cod, dis, hrsod_dhu, massive, general_bb_swin_v1_tiny, general
    #[arg(long, global = true, default_value = "general_bb_swin_v1_tiny")]
    pub variant: String,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, global = true, default_value = "fp16")]
    pub dtype: DType,

    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

    /// num dry run
    #[arg(long, global = true, default_value_t = 3)]
    pub num_dry_run: usize,
}

pub fn config(args: &BiRefNetArgs) -> Result<Config> {
    let config = if let Some(model) = &args.model {
        Config::birefnet().with_model_file(model)
    } else {
        match args.variant.as_str() {
            "cod" => Config::birefnet_cod(),
            "dis" => Config::birefnet_dis(),
            "hrsod_dhu" => Config::birefnet_hrsod_dhu(),
            "massive" => Config::birefnet_massive(),
            "general_bb_swin_v1_tiny" => Config::birefnet_general_bb_swin_v1_tiny(),
            "general" => Config::birefnet_general(),
            "hr_general" => Config::birefnet_hr_general(),
            "lite_general_2k" => Config::birefnet_lite_general_2k(),
            "portrait" => Config::birefnet_portrait(),
            "matting" => Config::birefnet_matting(),
            "hr_matting" => Config::birefnet_hr_matting(),
            _ => anyhow::bail!("Unsupported BiRefNet variant: {}", args.variant),
        }
    }
    .with_dtype_all(args.dtype)
    .with_device_all(args.device)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
