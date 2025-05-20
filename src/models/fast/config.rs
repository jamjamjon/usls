/// Model configuration for [FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation](https://github.com/czczup/FAST)
impl crate::Config {
    pub fn fast() -> Self {
        Self::db()
            .with_name("fast")
            .with_image_mean(&[0.798, 0.785, 0.772])
            .with_image_std(&[0.264, 0.2749, 0.287])
    }

    pub fn fast_tiny() -> Self {
        Self::fast().with_model_file("felixdittrich92-rep-tiny.onnx")
    }

    pub fn fast_small() -> Self {
        Self::fast().with_model_file("felixdittrich92-rep-small.onnx")
    }

    pub fn fast_base() -> Self {
        Self::fast().with_model_file("felixdittrich92-rep-base.onnx")
    }
}
