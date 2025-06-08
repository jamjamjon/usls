/// Model configuration for `MediaPipeSegmenter`
impl crate::Config {
    pub fn mediapipe() -> Self {
        Self::default()
            .with_name("mediapipe")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 256.into())
            .with_model_ixx(0, 3, 256.into())
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_normalize(true)
    }

    pub fn mediapipe_selfie_segmentater() -> Self {
        Self::mediapipe().with_model_file("selfie-segmenter.onnx")
    }

    pub fn mediapipe_selfie_segmentater_landscape() -> Self {
        Self::mediapipe().with_model_file("selfie-segmenter-landscape.onnx")
    }
}
