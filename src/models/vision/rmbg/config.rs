///
/// > # RMBG: BRIA Background Removal
/// >
/// > Professional background removal model by BRIA AI for high-quality image matting.
/// >
/// > # Paper & Code
/// >
/// > - **HuggingFace**: [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
/// >
/// > # Model Variants
/// >
/// > - **rmbg-1.4**: Version 1.4 background removal model
/// > - **rmbg-2.0**: Version 2.0 enhanced background removal model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Background Removal**: Professional-grade background segmentation
/// > - [X] **High-Quality Matting**: Clean edge detection and alpha matting
/// > - [X] **Commercial Use**: Optimized for commercial applications
/// >
/// Model configuration for `RMBG`
///
impl crate::Config {
    /// Base configuration for RMBG models
    pub fn rmbg() -> Self {
        Self::default()
            .with_name("rmbg")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 1024)
            .with_model_ixx(0, 3, 1024)
    }

    /// Version 1.4 background removal model
    pub fn rmbg1_4() -> Self {
        Self::rmbg()
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([1., 1., 1.])
            .with_model_file("1.4.onnx")
    }

    /// Version 2.0 enhanced background removal model
    pub fn rmbg2_0() -> Self {
        Self::rmbg()
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_model_file("2.0.onnx")
    }
}
