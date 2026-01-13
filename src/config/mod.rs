mod r#impl;
mod impl_ep;
mod impl_image_processor;
mod impl_inference;
mod impl_modules;
mod impl_ort;
#[cfg(feature = "vlm")]
mod impl_text_processor;
mod inference_params;
mod module;

pub use inference_params::*;
pub use module::*;
pub use r#impl::*;

/// Trait for types that can be constructed from a configuration.
///
/// This trait provides a standardized way to create instances from configuration
/// objects, enabling a consistent initialization pattern across the library.
pub trait FromConfig: Sized {
    /// The configuration type used to construct this type
    type Config;

    /// Create an instance from the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration object containing initialization parameters
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the constructed instance or an error
    /// if configuration is invalid or construction fails
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The configuration contains invalid parameters
    /// - Required resources cannot be loaded
    /// - The construction process encounters any other failure
    fn from_config(config: Self::Config) -> anyhow::Result<Self>;
}
