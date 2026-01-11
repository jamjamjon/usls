use anyhow::Result;
use clap::Parser;
use usls::{Config, DType};

#[derive(Parser, Debug)]
pub struct DepthProArgs {
    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub device: String,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: String,

    /// Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "q4f16")]
    pub dtype: DType,
}

pub fn config(args: &DepthProArgs) -> Result<Config> {
    let config = Config::depth_pro()
        .with_model_dtype(args.dtype)
        .with_device_all(args.device.parse()?)
        .with_image_processor_device(args.processor_device.parse()?);

    Ok(config)
}
