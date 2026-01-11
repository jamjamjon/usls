///
/// > # CLIP: Contrastive Language-Image Pre-Training
/// >
/// > Neural network trained on (image, text) pairs with zero-shot capabilities for computer vision tasks.
/// >
/// > # Paper & Code
/// >
/// > - **OpenAI CLIP**: [openai/CLIP](https://github.com/openai/CLIP)
/// > - **Jina CLIP v1**: [jinaai/jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1)
/// > - **Jina CLIP v2**: [jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2)
/// > - **MobileCLIP**: [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip)
/// > - **Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
/// >
/// > # Model Variants
/// >
/// > - **clip-vit-b16**: ViT-B/16 model for general image-text tasks
/// > - **clip-vit-b32**: ViT-B/32 model for general image-text tasks
/// > - **clip-vit-l14**: ViT-L/14 model for general image-text tasks
/// > - **jina-clip-v1**: Jina CLIP v1 with improved performance
/// > - **jina-clip-v2**: Jina CLIP v2 with 512x512 resolution
/// > - **mobileclip-s0/s1/s2/b/blt**: MobileCLIP variants for mobile devices
/// > - **mobileclip2-s0/s2/s3/s4/b/l14**: MobileCLIP v2 variants
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Zero-Shot Classification**: Classify images without training data
/// > - [X] **Image-Text Retrieval**: Retrieve relevant text for images
/// > - [X] **Text-Image Retrieval**: Retrieve relevant images for text
/// > - [X] **Mobile Optimization**: Lightweight models for mobile devices
/// > - [X] **Multi-Scale Support**: Various input resolutions
/// >
/// Model configuration for `CLIP`
///
impl crate::Config {
    /// Base configuration for CLIP models
    pub fn clip() -> Self {
        Self::default()
            .with_name("clip")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3)
            .with_visual_ixx(0, 2, 224)
            .with_visual_ixx(0, 3, 224)
            .with_image_mean([0.48145466, 0.4578275, 0.40821073])
            .with_image_std([0.26862954, 0.2613026, 0.2757771])
            .with_model_max_length(77)
            .with_tokenizer_file("clip/tokenizer.json")
            .with_tokenizer_config_file("clip/tokenizer_config.json")
            .with_special_tokens_map_file("clip/special_tokens_map.json")
            .with_config_file("clip/config.json")
    }

    /// ViT-B/16 model for general image-text tasks
    pub fn clip_vit_b16() -> Self {
        Self::clip()
            .with_visual_file("vit-b16-visual.onnx")
            .with_textual_file("vit-b16-textual.onnx")
    }

    /// ViT-B/32 model for general image-text tasks
    pub fn clip_vit_b32() -> Self {
        Self::clip()
            .with_visual_file("vit-b32-visual.onnx")
            .with_textual_file("vit-b32-textual.onnx")
    }

    /// ViT-L/14 model for general image-text tasks
    pub fn clip_vit_l14() -> Self {
        Self::clip()
            .with_visual_file("vit-l14-visual.onnx")
            .with_textual_file("vit-l14-textual.onnx")
    }

    fn jina_clip() -> Self {
        Self::default()
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3)
            .with_visual_ixx(0, 2, 224)
            .with_visual_ixx(0, 3, 224)
            .with_image_mean([0.48145466, 0.4578275, 0.40821073])
            .with_image_std([0.26862954, 0.2613026, 0.2757771])
            .with_visual_file("visual.onnx")
            .with_textual_file("textual.onnx")
    }

    /// Jina CLIP v1 with improved performance
    pub fn jina_clip_v1() -> Self {
        Self::jina_clip()
            .with_name("jina-clip-v1")
            .with_tokenizer_file("jina-clip-v1/tokenizer.json")
            .with_tokenizer_config_file("jina-clip-v1/tokenizer_config.json")
            .with_special_tokens_map_file("jina-clip-v1/special_tokens_map.json")
            .with_config_file("jina-clip-v1/config.json")
    }

    /// Jina CLIP v2 with 512x512 resolution
    pub fn jina_clip_v2() -> Self {
        Self::jina_clip()
            .with_name("jina-clip-v2")
            .with_visual_ixx(0, 2, 512)
            .with_visual_ixx(0, 3, 512)
            .with_tokenizer_file("jina-clip-v2/tokenizer.json")
            .with_tokenizer_config_file("jina-clip-v2/tokenizer_config.json")
            .with_special_tokens_map_file("jina-clip-v2/special_tokens_map.json")
            .with_config_file("jina-clip-v2/config.json")
    }

    fn mobileclip() -> Self {
        Self::default()
            .with_name("mobileclip")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3)
            .with_visual_ixx(0, 2, 224)
            .with_visual_ixx(0, 3, 224)
            .with_model_max_length(77)
            .with_tokenizer_file("clip/tokenizer.json")
            .with_tokenizer_config_file("clip/tokenizer_config.json")
            .with_special_tokens_map_file("clip/special_tokens_map.json")
    }

    /// MobileCLIP small model s0
    pub fn mobileclip_s0() -> Self {
        Self::mobileclip()
            .with_textual_file("s0-textual.onnx")
            .with_visual_file("s0-visual.onnx")
    }

    /// MobileCLIP small model s1
    pub fn mobileclip_s1() -> Self {
        Self::mobileclip()
            .with_textual_file("s1-textual.onnx")
            .with_visual_file("s1-visual.onnx")
    }

    /// MobileCLIP small model s2
    pub fn mobileclip_s2() -> Self {
        Self::mobileclip()
            .with_textual_file("s2-textual.onnx")
            .with_visual_file("s2-visual.onnx")
    }

    /// MobileCLIP base model
    pub fn mobileclip_b() -> Self {
        Self::mobileclip()
            .with_textual_file("b-textual.onnx")
            .with_visual_file("b-visual.onnx")
    }

    /// MobileCLIP base large transformer model
    pub fn mobileclip_blt() -> Self {
        Self::mobileclip()
            .with_textual_file("blt-textual.onnx")
            .with_visual_file("blt-visual.onnx")
    }

    fn mobileclip2() -> Self {
        Self::mobileclip().with_name("mobileclip2")
    }

    /// MobileCLIP v2 small model s0
    pub fn mobileclip2_s0() -> Self {
        Self::mobileclip2()
            .with_textual_file("s0-textual.onnx")
            .with_visual_file("s0-visual.onnx")
    }

    /// MobileCLIP v2 small model s2
    pub fn mobileclip2_s2() -> Self {
        Self::mobileclip2()
            .with_textual_file("s2-textual.onnx")
            .with_visual_file("s2-visual.onnx")
    }

    /// MobileCLIP v2 small model s3
    pub fn mobileclip2_s3() -> Self {
        Self::mobileclip2()
            .with_textual_file("s3-textual.onnx")
            .with_visual_file("s3-visual.onnx")
    }

    /// MobileCLIP v2 small model s4
    pub fn mobileclip2_s4() -> Self {
        Self::mobileclip2()
            .with_textual_file("s4-textual.onnx")
            .with_visual_file("s4-visual.onnx")
    }

    /// MobileCLIP v2 base model
    pub fn mobileclip2_b() -> Self {
        Self::mobileclip2()
            .with_textual_file("b-textual.onnx")
            .with_visual_file("b-visual.onnx")
    }

    /// MobileCLIP v2 large model
    pub fn mobileclip2_l14() -> Self {
        Self::mobileclip2()
            .with_textual_file("l-14-textual.onnx")
            .with_visual_file("l-14-visual.onnx")
    }
}
