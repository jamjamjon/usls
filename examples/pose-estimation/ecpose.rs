use anyhow::Result;
use clap::Args;
use usls::{Config, DType, Device, Scale};

#[derive(Args, Debug)]
pub struct EcposeArgs {
    /// Scale: s, m, l, x
    #[arg(long, default_value = "s")]
    pub scale: Scale,

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

pub fn config(args: &EcposeArgs) -> Result<Config> {
    let mut config = match args.scale {
        Scale::S => Config::ecpose_s(),
        Scale::M => Config::ecpose_m(),
        Scale::L => Config::ecpose_l(),
        Scale::X => Config::ecpose_x(),
        _ => anyhow::bail!(
            "Unsupported ECPose scale: {} (expected s, m, l, or x)",
            args.scale
        ),
    };

    config = if let Some(model) = &args.model {
        config.with_model_file(select_local_model_path(model, args.dtype))
    } else {
        config.with_model_dtype(args.dtype)
    };

    Ok(config
        .with_model_device(args.device)
        .with_model_batch_size_min_opt_max(args.min_batch, args.batch, args.max_batch)
        .with_model_num_dry_run(args.num_dry_run)
        .with_image_processor_device(args.processor_device))
}

fn select_local_model_path(path: &str, dtype: DType) -> String {
    if matches!(dtype, DType::Auto | DType::Fp32) {
        return path.to_string();
    }

    let Some(stem) = path.strip_suffix(".onnx") else {
        return path.to_string();
    };

    for suffix in [
        "-fp16", "_fp16", ".fp16", "-q4f16", "_q4f16", ".q4f16", "-q8", "_q8", ".q8", "-bnb4",
        "_bnb4", ".bnb4",
    ] {
        if let Some(base) = stem.strip_suffix(suffix) {
            return format!("{base}-{dtype}.onnx");
        }
    }

    format!("{stem}-{dtype}.onnx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_model_path_uses_dtype_suffix_once() {
        assert_eq!(
            select_local_model_path("/tmp/ecpose.onnx", DType::Fp16),
            "/tmp/ecpose-fp16.onnx"
        );
        assert_eq!(
            select_local_model_path("/tmp/ecpose-fp16.onnx", DType::Fp16),
            "/tmp/ecpose-fp16.onnx"
        );
    }
}
