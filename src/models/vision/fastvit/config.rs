use crate::NAMES_IMAGENET_1K;

///
/// > # FastViT: A Fast Hybrid Vision Transformer
/// >
/// > Fast hybrid vision transformer using structural reparameterization for efficient inference.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [apple/ml-fastvit](https://github.com/apple/ml-fastvit)
/// > - **Paper**: [FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/abs/2303.14189)
/// >
/// > # Model Variants
/// >
/// > - **fastvit-t8**: Tiny model with 8 layers
/// > - **fastvit-t8-distill**: Tiny model with knowledge distillation
/// > - **fastvit-t12**: Tiny model with 12 layers
/// > - **fastvit-t12-distill**: Tiny model with 12 layers + distillation
/// > - **fastvit-s12**: Small model with 12 layers
/// > - **fastvit-s12-distill**: Small model with 12 layers + distillation
/// > - **fastvit-sa12**: Small-Apple model with 12 layers
/// > - **fastvit-sa12-distill**: Small-Apple model with 12 layers + distillation
/// > - **fastvit-sa24**: Small-Apple model with 24 layers
/// > - **fastvit-sa24-distill**: Small-Apple model with 24 layers + distillation
/// > - **fastvit-sa36**: Small-Apple model with 36 layers
/// > - **fastvit-sa36-distill**: Small-Apple model with 36 layers + distillation
/// > - **fastvit-ma36**: Medium-Apple model with 36 layers
/// > - **fastvit-ma36-distill**: Medium-Apple model with 36 layers + distillation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image Classification**: 1000-class ImageNet classification
/// > - [X] **Structural Reparameterization**: Efficient CNN-Transformer hybrid
/// > - [X] **Knowledge Distillation**: Enhanced performance variants
/// >
/// Model configuration for `FastViT`
///
impl crate::Config {
    /// Base configuration for FastViT models
    pub fn fastvit() -> Self {
        Self::default()
            .with_name("fastvit")
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

    /// Tiny model with 8 layers
    pub fn fastvit_t8() -> Self {
        Self::fastvit().with_model_file("t8.onnx")
    }

    /// Tiny model with knowledge distillation
    pub fn fastvit_t8_distill() -> Self {
        Self::fastvit().with_model_file("t8-distill.onnx")
    }

    /// Tiny model with 12 layers
    pub fn fastvit_t12() -> Self {
        Self::fastvit().with_model_file("t12.onnx")
    }

    /// Tiny model with 12 layers + distillation
    pub fn fastvit_t12_distill() -> Self {
        Self::fastvit().with_model_file("t12-distill.onnx")
    }

    /// Small model with 12 layers
    pub fn fastvit_s12() -> Self {
        Self::fastvit().with_model_file("s12.onnx")
    }

    /// Small model with 12 layers + distillation
    pub fn fastvit_s12_distill() -> Self {
        Self::fastvit().with_model_file("s12-distill.onnx")
    }

    /// Small-Apple model with 12 layers
    pub fn fastvit_sa12() -> Self {
        Self::fastvit().with_model_file("sa12.onnx")
    }

    /// Small-Apple model with 12 layers + distillation
    pub fn fastvit_sa12_distill() -> Self {
        Self::fastvit().with_model_file("sa12-distill.onnx")
    }

    /// Small-Apple model with 24 layers
    pub fn fastvit_sa24() -> Self {
        Self::fastvit().with_model_file("sa24.onnx")
    }

    /// Small-Apple model with 24 layers + distillation
    pub fn fastvit_sa24_distill() -> Self {
        Self::fastvit().with_model_file("sa24-distill.onnx")
    }

    /// Small-Apple model with 36 layers
    pub fn fastvit_sa36() -> Self {
        Self::fastvit().with_model_file("sa36.onnx")
    }

    /// Small-Apple model with 36 layers + distillation
    pub fn fastvit_sa36_distill() -> Self {
        Self::fastvit().with_model_file("sa36-distill.onnx")
    }

    /// Medium-Apple model with 36 layers
    pub fn fastvit_ma36() -> Self {
        Self::fastvit().with_model_file("ma36.onnx")
    }

    /// Medium-Apple model with 36 layers + distillation
    pub fn fastvit_ma36_distill() -> Self {
        Self::fastvit().with_model_file("ma36-distill.onnx")
    }
}
