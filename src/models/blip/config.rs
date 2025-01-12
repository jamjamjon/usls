/// Model configuration for `BLIP`
impl crate::Options {
    pub fn blip() -> Self {
        Self::default().with_model_name("blip").with_batch_size(1)
    }

    #[allow(clippy::excessive_precision)]
    pub fn blip_visual() -> Self {
        Self::blip()
            .with_model_kind(crate::Kind::Vision)
            .with_model_ixx(0, 2, 384.into())
            .with_model_ixx(0, 3, 384.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.26130258, 0.27577711])
            .with_resize_filter("Bilinear")
            .with_normalize(true)
    }

    pub fn blip_textual() -> Self {
        Self::blip().with_model_kind(crate::Kind::Language)
    }

    pub fn blip_v1_base_caption_visual() -> Self {
        Self::blip_visual()
            .with_model_version(1.0.into())
            .with_model_file("v1-base-caption-visual.onnx")
    }

    pub fn blip_v1_base_caption_textual() -> Self {
        Self::blip_textual()
            .with_model_version(1.0.into())
            .with_model_file("v1-base-caption-textual.onnx")
    }
}
