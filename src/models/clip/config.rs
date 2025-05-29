/// Model configuration for `CLIP`
impl crate::Config {
    pub fn clip() -> Self {
        Self::default()
            .with_name("clip")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3.into())
            .with_visual_ixx(0, 2, 224.into())
            .with_visual_ixx(0, 3, 224.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.2613026, 0.2757771])
            .with_model_max_length(77)
            .with_tokenizer_file("clip/tokenizer.json")
            .with_tokenizer_config_file("clip/tokenizer_config.json")
            .with_special_tokens_map_file("clip/special_tokens_map.json")
            .with_config_file("clip/config.json")
    }

    pub fn clip_vit_b16() -> Self {
        Self::clip()
            .with_visual_file("vit-b16-visual.onnx")
            .with_textual_file("vit-b16-textual.onnx")
    }

    pub fn clip_vit_b32() -> Self {
        Self::clip()
            .with_visual_file("vit-b32-visual.onnx")
            .with_textual_file("vit-b32-textual.onnx")
    }

    pub fn clip_vit_l14() -> Self {
        Self::clip()
            .with_visual_file("vit-l14-visual.onnx")
            .with_textual_file("vit-l14-textual.onnx")
    }

    pub fn jina_clip() -> Self {
        Self::default()
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3.into())
            .with_visual_ixx(0, 2, 224.into())
            .with_visual_ixx(0, 3, 224.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.2613026, 0.2757771])
            .with_visual_file("visual.onnx")
            .with_textual_file("textual.onnx")
    }

    pub fn jina_clip_v1() -> Self {
        Self::jina_clip()
            .with_name("jina-clip-v1")
            .with_tokenizer_file("jina-clip-v1/tokenizer.json")
            .with_tokenizer_config_file("jina-clip-v1/tokenizer_config.json")
            .with_special_tokens_map_file("jina-clip-v1/special_tokens_map.json")
            .with_config_file("jina-clip-v1/config.json")
    }

    pub fn jina_clip_v2() -> Self {
        Self::jina_clip()
            .with_name("jina-clip-v2")
            .with_visual_ixx(0, 2, 512.into())
            .with_visual_ixx(0, 3, 512.into())
            .with_tokenizer_file("jina-clip-v2/tokenizer.json")
            .with_tokenizer_config_file("jina-clip-v2/tokenizer_config.json")
            .with_special_tokens_map_file("jina-clip-v2/special_tokens_map.json")
            .with_config_file("jina-clip-v2/config.json")
    }

    pub fn mobileclip() -> Self {
        Self::default()
            .with_name("mobileclip")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3.into())
            .with_visual_ixx(0, 2, 224.into())
            .with_visual_ixx(0, 3, 224.into())
            .with_model_max_length(77)
            .with_tokenizer_file("clip/tokenizer.json")
            .with_tokenizer_config_file("clip/tokenizer_config.json")
            .with_special_tokens_map_file("clip/special_tokens_map.json")
    }

    pub fn mobileclip_s0() -> Self {
        Self::mobileclip()
            .with_textual_file("s0-textual.onnx")
            .with_visual_file("s0-visual.onnx")
    }

    pub fn mobileclip_s1() -> Self {
        Self::mobileclip()
            .with_textual_file("s1-textual.onnx")
            .with_visual_file("s1-visual.onnx")
    }

    pub fn mobileclip_s2() -> Self {
        Self::mobileclip()
            .with_textual_file("s2-textual.onnx")
            .with_visual_file("s2-visual.onnx")
    }

    pub fn mobileclip_b() -> Self {
        Self::mobileclip()
            .with_textual_file("b-textual.onnx")
            .with_visual_file("b-visual.onnx")
    }

    pub fn mobileclip_blt() -> Self {
        Self::mobileclip()
            .with_textual_file("blt-textual.onnx")
            .with_visual_file("blt-visual.onnx")
    }
}
