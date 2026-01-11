///
/// > # MODNet: Trimap-Free Portrait Matting in Real Time
/// >
/// > Real-time portrait matting without requiring trimap input for background removal.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)
/// > - **Paper**: [MODNet: Trimap-Free Portrait Matting in Real Time](https://arxiv.org/abs/2011.11961)
/// >
/// > # Model Variants
/// >
/// > - **modnet-photographic**: Photographic portrait matting model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Portrait Matting**: High-quality portrait background removal
/// > - [X] **Trimap-Free**: No manual trimap annotation required
/// > - [X] **Real-time Processing**: Fast inference for real-time applications
/// >
/// Model configuration for `MODNet`
///
impl crate::Config {
    /// Base configuration for MODNet models
    pub fn modnet() -> Self {
        Self::default()
            .with_name("modnet")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 2, (416, 512, 800))
            .with_model_ixx(0, 3, (416, 512, 800))
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_normalize(true)
    }

    /// Photographic portrait matting model
    pub fn modnet_photographic() -> Self {
        Self::modnet().with_model_file("photographic-portrait-matting.onnx")
    }
}
