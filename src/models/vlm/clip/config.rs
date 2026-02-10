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
/// > ## OpenAI CLIP
/// > - **clip-vit-b16**: ViT-B/16 (85M params)
/// > - **clip-vit-b32**: ViT-B/32 (87M params)
/// > - **clip-vit-l14**: ViT-L/14 (304M params)
/// >
/// > ## Jina CLIP
/// > - **jina-clip-v1**: Improved performance, 224x224
/// > - **jina-clip-v2**: 512x512 resolution, better accuracy
/// >
/// > ## MobileCLIP (Apple)
/// > - **mobileclip-s0/s1/s2**: Small variants (0-2)
/// > - **mobileclip-b**: Base variant
/// > - **mobileclip-blt**: Base with large text encoder
/// >
/// > ## MobileCLIP v2
/// > - **mobileclip2-s0/s2/s4/b/l14**: Enhanced mobile variants
/// >
/// > ## SigLIP (Google DeepMind)
/// > - **siglip-b16-224/256/384/512**: Base models, patch16
/// > - **siglip-l16-256/384**: Large models, patch16
/// >
/// > ## SigLIP v2 (Google DeepMind)
/// > - **siglip2-b16-224/256/384/512**: Base models v2
/// > - **siglip2-l16-256/384/512**: Large models v2
/// > - **siglip2-so400m-patch14/16**: 400M parameter models
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Zero-Shot Classification**: Classify images without training data
/// > - [X] **Image-Text Retrieval**: Retrieve relevant text for images
/// > - [X] **Text-Image Retrieval**: Retrieve relevant images for text
/// > - [X] **Mobile Optimization**: Lightweight models for mobile devices
/// > - [X] **Multi-Scale Support**: Various input resolutions (224, 256, 384, 512)
/// > - [X] **Dual Encoder Architecture**: Separate vision and text encoders
/// > - [X] **Contrastive Learning**: Image-text similarity scoring
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

    pub fn siglip() -> Self {
        Self::clip()
            .with_name("clip")
            .with_batch_size_min_opt_max_all(1, 1, 8) // batch size
            .with_visual_ixx(0, 1, 3) // channel
            .with_textual_ixx(0, 1, 64) // seq len
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_model_max_length(64)
    }

    /// SigLIP Base, patch16, 224x224
    pub fn siglip_b16_224() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 224)
            .with_visual_ixx(0, 3, 224)
            .with_tokenizer_file("Xenova/siglip-base-patch16-224/tokenizer.json")
            .with_tokenizer_config_file("Xenova/siglip-base-patch16-224/tokenizer_config.json")
            .with_special_tokens_map_file("Xenova/siglip-base-patch16-224/special_tokens_map.json")
            .with_textual_file("Xenova/siglip-base-patch16-224/onnx/text_model.onnx")
            .with_visual_file("Xenova/siglip-base-patch16-224/onnx/vision_model.onnx")
    }

    /// SigLIP Base, patch16, 256x256
    pub fn siglip_b16_256() -> Self {
        Self::siglip()
            .with_textual_file("Xenova/siglip-base-patch16-256/onnx/text_model.onnx")
            .with_visual_file("Xenova/siglip-base-patch16-256/onnx/vision_model.onnx")
            .with_visual_ixx(0, 2, 256)
            .with_visual_ixx(0, 3, 256)
            .with_tokenizer_file("Xenova/siglip-base-patch16-256/tokenizer.json")
            .with_tokenizer_config_file("Xenova/siglip-base-patch16-256/tokenizer_config.json")
            .with_special_tokens_map_file("Xenova/siglip-base-patch16-256/special_tokens_map.json")
    }

    /// SigLIP Base, patch16, 384x384
    pub fn siglip_b16_384() -> Self {
        Self::siglip()
            .with_textual_file("Xenova/siglip-base-patch16-384/onnx/text_model.onnx")
            .with_visual_file("Xenova/siglip-base-patch16-384/onnx/vision_model.onnx")
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_tokenizer_file("Xenova/siglip-base-patch16-384/tokenizer.json")
            .with_tokenizer_config_file("Xenova/siglip-base-patch16-384/tokenizer_config.json")
            .with_special_tokens_map_file("Xenova/siglip-base-patch16-384/special_tokens_map.json")
    }

    /// SigLIP Base, patch16, 512x512
    pub fn siglip_b16_512() -> Self {
        Self::siglip()
            .with_textual_file("Xenova/siglip-base-patch16-512/onnx/text_model.onnx")
            .with_visual_file("Xenova/siglip-base-patch16-512/onnx/vision_model.onnx")
            .with_visual_ixx(0, 2, 512)
            .with_visual_ixx(0, 3, 512)
            .with_tokenizer_file("Xenova/siglip-base-patch16-512/tokenizer.json")
            .with_tokenizer_config_file("Xenova/siglip-base-patch16-512/tokenizer_config.json")
            .with_special_tokens_map_file("Xenova/siglip-base-patch16-512/special_tokens_map.json")
    }

    /// SigLIP Large, patch16, 256x256
    pub fn siglip_l16_256() -> Self {
        Self::siglip()
            .with_textual_file("Xenova/siglip-large-patch16-256/onnx/text_model.onnx")
            .with_visual_file("Xenova/siglip-large-patch16-256/onnx/vision_model.onnx")
            .with_visual_ixx(0, 2, 256)
            .with_visual_ixx(0, 3, 256)
            .with_tokenizer_file("Xenova/siglip-large-patch16-256/tokenizer.json")
            .with_tokenizer_config_file("Xenova/siglip-large-patch16-256/tokenizer_config.json")
            .with_special_tokens_map_file("Xenova/siglip-large-patch16-256/special_tokens_map.json")
    }

    /// SigLIP Large, patch16, 384x384
    pub fn siglip_l16_384() -> Self {
        Self::siglip()
            .with_textual_file("Xenova/siglip-large-patch16-384/onnx/text_model.onnx")
            .with_visual_file("Xenova/siglip-large-patch16-384/onnx/vision_model.onnx")
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_tokenizer_file("Xenova/siglip-large-patch16-384/tokenizer.json")
            .with_tokenizer_config_file("Xenova/siglip-large-patch16-384/tokenizer_config.json")
            .with_special_tokens_map_file("Xenova/siglip-large-patch16-384/special_tokens_map.json")
    }

    /// SigLIP v2 Base, patch16, 224x224
    pub fn siglip2_b16_224() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 224)
            .with_visual_ixx(0, 3, 224)
            .with_tokenizer_file("onnx-community/siglip2-base-patch16-224-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-base-patch16-224-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-base-patch16-224-ONNX/special_tokens_map.json",
            )
            .with_textual_file("onnx-community/siglip2-base-patch16-224-ONNX/onnx/text_model.onnx")
            .with_visual_file("onnx-community/siglip2-base-patch16-224-ONNX/onnx/vision_model.onnx")
    }

    /// SigLIP v2 Base, patch16, 256x256
    pub fn siglip2_b16_256() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 256)
            .with_visual_ixx(0, 3, 256)
            .with_tokenizer_file("onnx-community/siglip2-base-patch16-256-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-base-patch16-256-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-base-patch16-256-ONNX/special_tokens_map.json",
            )
            .with_textual_file("onnx-community/siglip2-base-patch16-256-ONNX/onnx/text_model.onnx")
            .with_visual_file("onnx-community/siglip2-base-patch16-256-ONNX/onnx/vision_model.onnx")
    }

    /// SigLIP v2 Base, patch16, 384x384
    pub fn siglip2_b16_384() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_tokenizer_file("onnx-community/siglip2-base-patch16-384-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-base-patch16-384-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-base-patch16-384-ONNX/special_tokens_map.json",
            )
            .with_textual_file("onnx-community/siglip2-base-patch16-384-ONNX/onnx/text_model.onnx")
            .with_visual_file("onnx-community/siglip2-base-patch16-384-ONNX/onnx/vision_model.onnx")
    }

    /// SigLIP v2 Base, patch16, 512x512
    pub fn siglip2_b16_512() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 512)
            .with_visual_ixx(0, 3, 512)
            .with_tokenizer_file("onnx-community/siglip2-base-patch16-512-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-base-patch16-512-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-base-patch16-512-ONNX/special_tokens_map.json",
            )
            .with_textual_file("onnx-community/siglip2-base-patch16-512-ONNX/onnx/text_model.onnx")
            .with_visual_file("onnx-community/siglip2-base-patch16-512-ONNX/onnx/vision_model.onnx")
    }

    /// SigLIP v2 Large, patch16, 256x256
    pub fn siglip2_l16_256() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 256)
            .with_visual_ixx(0, 3, 256)
            .with_tokenizer_file("onnx-community/siglip2-large-patch16-256-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-large-patch16-256-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-large-patch16-256-ONNX/special_tokens_map.json",
            )
            .with_textual_file("onnx-community/siglip2-large-patch16-256-ONNX/onnx/text_model.onnx")
            .with_visual_file(
                "onnx-community/siglip2-large-patch16-256-ONNX/onnx/vision_model.onnx",
            )
    }

    /// SigLIP v2 Large, patch16, 384x384
    pub fn siglip2_l16_384() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_tokenizer_file("onnx-community/siglip2-large-patch16-384-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-large-patch16-384-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-large-patch16-384-ONNX/special_tokens_map.json",
            )
            .with_textual_file("onnx-community/siglip2-large-patch16-384-ONNX/onnx/text_model.onnx")
            .with_visual_file(
                "onnx-community/siglip2-large-patch16-384-ONNX/onnx/vision_model.onnx",
            )
    }

    /// SigLIP v2 Large, patch16, 512x512
    pub fn siglip2_l16_512() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 512)
            .with_visual_ixx(0, 3, 512)
            .with_tokenizer_file("onnx-community/siglip2-large-patch16-512-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-large-patch16-512-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-large-patch16-512-ONNX/special_tokens_map.json",
            )
            .with_textual_file("onnx-community/siglip2-large-patch16-512-ONNX/onnx/text_model.onnx")
            .with_visual_file(
                "onnx-community/siglip2-large-patch16-512-ONNX/onnx/vision_model.onnx",
            )
    }

    /// SigLIP v2 400M, patch14, 224x224
    pub fn siglip2_so400m_patch14_224() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 224)
            .with_visual_ixx(0, 3, 224)
            .with_tokenizer_file("onnx-community/siglip2-so400m-patch14-224-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-so400m-patch14-224-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-so400m-patch14-224-ONNX/special_tokens_map.json",
            )
            .with_textual_file(
                "onnx-community/siglip2-so400m-patch14-224-ONNX/onnx/text_model.onnx",
            )
            .with_visual_file(
                "onnx-community/siglip2-so400m-patch14-224-ONNX/onnx/vision_model.onnx",
            )
    }

    /// SigLIP v2 400M, patch14, 384x384
    pub fn siglip2_so400m_patch14_384() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_tokenizer_file("onnx-community/siglip2-so400m-patch14-384-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-so400m-patch14-384-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-so400m-patch14-384-ONNX/special_tokens_map.json",
            )
            .with_textual_file(
                "onnx-community/siglip2-so400m-patch14-384-ONNX/onnx/text_model.onnx",
            )
            .with_visual_file(
                "onnx-community/siglip2-so400m-patch14-384-ONNX/onnx/vision_model.onnx",
            )
    }

    /// SigLIP v2 400M, patch16, 256x256
    pub fn siglip2_so400m_patch16_256() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 256)
            .with_visual_ixx(0, 3, 256)
            .with_tokenizer_file("onnx-community/siglip2-so400m-patch16-256-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-so400m-patch16-256-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-so400m-patch16-256-ONNX/special_tokens_map.json",
            )
            .with_textual_file(
                "onnx-community/siglip2-so400m-patch16-256-ONNX/onnx/text_model.onnx",
            )
            .with_visual_file(
                "onnx-community/siglip2-so400m-patch16-256-ONNX/onnx/vision_model.onnx",
            )
    }

    /// SigLIP v2 400M, patch16, 384x384
    pub fn siglip2_so400m_patch16_384() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_tokenizer_file("onnx-community/siglip2-so400m-patch16-384-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-so400m-patch16-384-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-so400m-patch16-384-ONNX/special_tokens_map.json",
            )
            .with_textual_file(
                "onnx-community/siglip2-so400m-patch16-384-ONNX/onnx/text_model.onnx",
            )
            .with_visual_file(
                "onnx-community/siglip2-so400m-patch16-384-ONNX/onnx/vision_model.onnx",
            )
    }

    /// SigLIP v2 400M, patch16, 512x512
    pub fn siglip2_so400m_patch16_512() -> Self {
        Self::siglip()
            .with_visual_ixx(0, 2, 512)
            .with_visual_ixx(0, 3, 512)
            .with_tokenizer_file("onnx-community/siglip2-so400m-patch16-512-ONNX/tokenizer.json")
            .with_tokenizer_config_file(
                "onnx-community/siglip2-so400m-patch16-512-ONNX/tokenizer_config.json",
            )
            .with_special_tokens_map_file(
                "onnx-community/siglip2-so400m-patch16-512-ONNX/special_tokens_map.json",
            )
            .with_textual_file(
                "onnx-community/siglip2-so400m-patch16-512-ONNX/onnx/text_model.onnx",
            )
            .with_visual_file(
                "onnx-community/siglip2-so400m-patch16-512-ONNX/onnx/vision_model.onnx",
            )
    }
}
