/// Model configuration for [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718)
impl crate::Config {
    pub fn linknet() -> Self {
        Self::fast()
            .with_name("linknet")
            .with_image_mean(&[0.798, 0.785, 0.772])
            .with_image_std(&[0.264, 0.2749, 0.287])
    }

    pub fn linknet_r18() -> Self {
        Self::linknet().with_model_file("felixdittrich92-r18.onnx")
    }

    pub fn linknet_r34() -> Self {
        Self::linknet().with_model_file("felixdittrich92-r34.onnx")
    }

    pub fn linknet_r50() -> Self {
        Self::linknet().with_model_file("felixdittrich92-r50.onnx")
    }
}
