///
/// > # MediaPipe: Selfie Segmentation Model
/// >
/// > Google's MediaPipe selfie segmentation model for portrait and background separation.
/// >
/// > # Paper & Code
/// >
/// > - **Official Guide**: [Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
/// >
/// > # Model Variants
/// >
/// > - **mediapipe-selfie-segmenter**: Standard portrait segmentation model
/// > - **mediapipe-selfie-segmenter-landscape**: Landscape-oriented portrait segmentation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Portrait Segmentation**: Selfie and portrait background separation
/// > - [X] **Real-time Performance**: Optimized for mobile and edge devices
/// >
/// Model configuration for `MediaPipeSegmenter`
///
impl crate::Config {
    /// Base configuration for MediaPipe models
    pub fn mediapipe() -> Self {
        Self::default()
            .with_name("mediapipe")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 256)
            .with_model_ixx(0, 3, 256)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_normalize(true)
    }

    /// Standard portrait segmentation model
    pub fn mediapipe_selfie_segmentater() -> Self {
        Self::mediapipe().with_model_file("selfie-segmenter.onnx")
    }

    /// Landscape-oriented portrait segmentation
    pub fn mediapipe_selfie_segmentater_landscape() -> Self {
        Self::mediapipe().with_model_file("selfie-segmenter-landscape.onnx")
    }
}
