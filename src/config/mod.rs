mod r#impl;
pub mod impl_all;
pub mod impl_ep;
pub mod impl_image_processor;
pub mod impl_inference;
pub mod impl_modules;
#[cfg(feature = "vlm")]
pub mod impl_text_processor;
mod inference_params;
mod macros;
mod module;

pub use inference_params::InferenceParams;
pub use module::Module;
pub use r#impl::Config;

pub trait FromConfig: Sized {
    type Config;

    fn from_config(config: Self::Config) -> anyhow::Result<Self>;
}
