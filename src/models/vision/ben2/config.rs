///
/// > # BEN2: Background Erase Network
/// >
/// > Background removal model for image segmentation and matting.
/// >
/// > # Paper & Code
/// >
/// > - **HuggingFace**: [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2)
/// >
/// > # Model Variants
/// >
/// > - **ben2-base**: Base background removal model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Background Removal**: Image background segmentation
/// >
/// Model configuration for `BEN2`
///
impl crate::Config {
    /// Base background removal model
    pub fn ben2_base() -> Self {
        Self::rmbg().with_model_file("ben2-base.onnx")
    }
}
