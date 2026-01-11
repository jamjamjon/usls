///
/// > # FAST: Faster Arbitrarily-Shaped Text Detector
/// >
/// > Fast arbitrarily-shaped text detector with minimalist kernel representation.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [czczup/FAST](https://github.com/czczup/FAST)
/// >
/// > # Model Variants
/// >
/// > - **fast-tiny**: Tiny model for real-time text detection
/// > - **fast-small**: Small model for balanced performance
/// > - **fast-base**: Base model for high accuracy
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Arbitrary-Shaped Text Detection**: Curved and irregular text detection
/// > - [X] **Minimalist Kernel Representation**: Efficient text detection approach
/// >
/// Model configuration for `FAST`
///
impl crate::Config {
    /// Base configuration for FAST models
    pub fn fast() -> Self {
        Self::db()
            .with_name("fast")
            .with_image_mean([0.798, 0.785, 0.772])
            .with_image_std([0.264, 0.2749, 0.287])
    }

    /// Tiny model for real-time text detection
    pub fn fast_tiny() -> Self {
        Self::fast().with_model_file("felixdittrich92-rep-tiny.onnx")
    }

    /// Small model for balanced performance
    pub fn fast_small() -> Self {
        Self::fast().with_model_file("felixdittrich92-rep-small.onnx")
    }

    /// Base model for high accuracy
    pub fn fast_base() -> Self {
        Self::fast().with_model_file("felixdittrich92-rep-base.onnx")
    }
}
