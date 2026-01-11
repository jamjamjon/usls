use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct DeimArgs {
    /// Scale: n, s, m, b, l
    #[arg(long, default_value = "s")]
    pub scale: Scale,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub dtype: DType,

    /// version: 1, 2
    #[arg(long, default_value = "2", value_parser = clap::value_parser!(u8).range(1..=2))]
    pub ver: u8,

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
}

pub fn config(args: &DeimArgs) -> Result<Config> {
    let mut config = match (args.ver, &args.scale) {
        (1, Scale::S) => Config::deim_dfine_s_coco(),
        (1, Scale::M) => Config::deim_dfine_m_coco(),
        (1, Scale::L) => Config::deim_dfine_l_coco(),
        (1, Scale::X) => Config::deim_dfine_x_coco(),
        (2, Scale::A) => Config::deim_v2_atto_coco(),
        (2, Scale::F) => Config::deim_v2_femto_coco(),
        (2, Scale::P) => Config::deim_v2_pico_coco(),
        (2, Scale::N) => Config::deim_v2_n_coco(),
        (2, Scale::S) => Config::deim_v2_s_coco(),
        (2, Scale::M) => Config::deim_v2_m_coco(),
        (2, Scale::L) => Config::deim_v2_l_coco(),
        (2, Scale::X) => Config::deim_v2_x_coco(),
        _ => anyhow::bail!(
            "Unsupported version {} with scale {}. Try v1/v2 with appropriate scales.",
            args.ver,
            args.scale
        ),
    }
    .with_batch_size_all_min_opt_max(args.min_batch, args.batch, args.max_batch)
    .with_model_device(args.device)
    .with_model_dtype(args.dtype)
    .with_num_dry_run_all(args.num_dry_run);

    if let Some(device) = args.processor_device {
        config = config.with_image_processor_device(device);
    }

    Ok(config)
}
