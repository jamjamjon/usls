#[macro_use]
mod macros;
mod config;
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
mod layout;
mod plan;
mod processor;
mod transform;

pub use config::*;
pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
pub use layout::*;
pub use plan::*;
pub use processor::*;
pub use transform::*;

pub trait TransformExecutor {
    fn execute_plan(
        &self,
        images: &[crate::Image],
        plan: &crate::ImagePlan,
    ) -> anyhow::Result<(crate::XAny, Vec<crate::ImageTransformInfo>)>;
}
