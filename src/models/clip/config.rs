use crate::Kind;

/// Model configuration for `CLIP`
impl crate::Options {
    pub fn clip() -> Self {
        Self::default()
            .with_model_name("clip")
            .with_model_ixx(0, 0, 1.into())
    }

    pub fn clip_visual() -> Self {
        Self::clip()
            .with_model_kind(Kind::Vision)
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.2613026, 0.2757771])
    }

    pub fn clip_textual() -> Self {
        Self::clip()
            .with_model_kind(Kind::Language)
            .with_model_max_length(77)
    }

    pub fn clip_vit_b16_visual() -> Self {
        Self::clip_visual().with_model_file("vit-b16-visual.onnx")
    }

    pub fn clip_vit_b16_textual() -> Self {
        Self::clip_textual().with_model_file("vit-b16-textual.onnx")
    }

    pub fn clip_vit_b32_visual() -> Self {
        Self::clip_visual().with_model_file("vit-b32-visual.onnx")
    }

    pub fn clip_vit_b32_textual() -> Self {
        Self::clip_textual().with_model_file("vit-b32-textual.onnx")
    }

    pub fn clip_vit_l14_visual() -> Self {
        Self::clip_visual().with_model_file("vit-l14-visual.onnx")
    }

    pub fn clip_vit_l14_textual() -> Self {
        Self::clip_textual().with_model_file("vit-l14-textual.onnx")
    }

    pub fn jina_clip_v1() -> Self {
        Self::default()
            .with_model_name("jina-clip-v1")
            .with_model_ixx(0, 0, 1.into())
    }

    pub fn jina_clip_v1_visual() -> Self {
        Self::jina_clip_v1()
            .with_model_kind(Kind::Vision)
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.2613026, 0.2757771])
            .with_model_file("visual.onnx")
    }

    pub fn jina_clip_v1_textual() -> Self {
        Self::jina_clip_v1()
            .with_model_kind(Kind::Language)
            .with_model_file("textual.onnx")
    }
}
