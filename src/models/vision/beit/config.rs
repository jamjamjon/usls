use crate::NAMES_IMAGENET_1K;

///
/// > # BEiT: BERT Pre-Training of Image Transformers
/// >
/// > Vision Transformer pre-trained using masked image modeling approach.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beit)
/// > - **Paper**: [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)
/// >
/// > # Model Variants
/// >
/// > - **beit-base**: Base-sized ViT-B/16 architecture
/// > - **beit-large**: Large-sized ViT-L/16 architecture
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image Classification**: 1000-class ImageNet classification
/// >
/// Model configuration for `BEiT`
///
impl crate::Config {
    /// Base configuration for BEiT models
    pub fn beit() -> Self {
        Self::default()
            .with_name("beit")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 224)
            .with_model_ixx(0, 3, 224)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_apply_softmax(true)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    /// Base model for ImageNet classification
    pub fn beit_base() -> Self {
        Self::beit().with_model_file("b.onnx")
    }

    /// Large model for ImageNet classification
    pub fn beit_large() -> Self {
        Self::beit().with_model_file("l.onnx")
    }
}
