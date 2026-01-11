use crate::NAMES_IMAGENET_1K;

///
/// > # DeiT: Data-Efficient Image Transformers
/// >
/// > Data-efficient Vision Transformer training for image classification without large datasets.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/deit](https://github.com/facebookresearch/deit)
/// > - **Paper**: [Training data-efficient image transformers & distillation](https://arxiv.org/abs/2012.12877)
/// >
/// > # Model Variants
/// >
/// > - **deit-tiny-distill**: Tiny model with distillation
/// > - **deit-small-distill**: Small model with distillation
/// > - **deit-base-distill**: Base model with distillation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image Classification**: 1000-class ImageNet classification
/// > - [X] **Knowledge Distillation**: Teacher-student training approach
/// >
/// Model configuration for `DeiT`
///
impl crate::Config {
    /// Base configuration for DeiT models
    pub fn deit() -> Self {
        Self::default()
            .with_name("deit")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 224)
            .with_model_ixx(0, 3, 224)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_apply_softmax(true)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    /// Tiny model with distillation
    pub fn deit_tiny_distill() -> Self {
        Self::deit().with_model_file("t-distill.onnx")
    }

    /// Small model with distillation
    pub fn deit_small_distill() -> Self {
        Self::deit().with_model_file("s-distill.onnx")
    }

    /// Base model with distillation
    pub fn deit_base_distill() -> Self {
        Self::deit().with_model_file("b-distill.onnx")
    }
}
