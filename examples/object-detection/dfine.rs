use anyhow::Result;
use clap::{Args, ValueEnum};
use usls::{Config, DType, Device};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Kind {
    #[clap(name = "n-coco")]
    N,
    #[clap(name = "s-coco")]
    S,
    #[clap(name = "m-coco")]
    M,
    #[clap(name = "l-coco")]
    L,
    #[clap(name = "x-coco")]
    X,
    #[clap(name = "s-obj365")]
    SObj365,
    #[clap(name = "m-obj365")]
    MObj365,
    #[clap(name = "l-obj365")]
    LObj365,
    #[clap(name = "x-obj365")]
    XObj365,
}

#[derive(Args, Debug)]
pub struct DfineArgs {
    /// Model variant
    #[arg(long, value_enum, default_value = "n-coco")]
    pub kind: Kind,

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

pub fn config(args: &DfineArgs) -> Result<Config> {
    let config = match args.kind {
        Kind::N => Config::d_fine_n_coco(),
        Kind::S => Config::d_fine_s_coco(),
        Kind::M => Config::d_fine_m_coco(),
        Kind::L => Config::d_fine_l_coco(),
        Kind::X => Config::d_fine_x_coco(),
        Kind::SObj365 => Config::d_fine_s_coco_obj365(),
        Kind::MObj365 => Config::d_fine_m_coco_obj365(),
        Kind::LObj365 => Config::d_fine_l_coco_obj365(),
        Kind::XObj365 => Config::d_fine_x_coco_obj365(),
    }
    .with_model_dtype(args.dtype)
    .with_model_device(args.device)
    .with_batch_size_min_opt_max_all(args.min_batch, args.batch, args.max_batch)
    .with_num_dry_run_all(args.num_dry_run)
    .with_image_processor_device(args.processor_device);

    Ok(config)
}
