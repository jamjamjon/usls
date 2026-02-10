#[macro_use]
mod macros;
mod config;
mod dtype;
mod engine;
mod engines;
mod ep_config;
mod iiix;
mod inputs;
mod min_opt_max;
#[allow(clippy::all)]
pub(crate) mod onnx;
mod x;
mod x_any;
mod xs;

pub use config::*;
pub use engine::*;
pub use engines::*;
pub use ep_config::*;
pub(crate) use iiix::*;
pub use inputs::*;
pub use min_opt_max::*;
pub use xs::*;

pub fn load_onnx<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<onnx::ModelProto> {
    use prost::Message;
    let path_ref = p.as_ref();
    let f = std::fs::read(path_ref).map_err(|err| {
        anyhow::anyhow!("Failed to read ONNX file '{path_ref:?}': {err}. Error: {err}")
    })?;
    onnx::ModelProto::decode(f.as_slice()).map_err(|err| {
        anyhow::anyhow!(
            "Failed to read the ONNX model: The file might be incomplete or corrupted. More detailed: {err}"
        )
    })
}
