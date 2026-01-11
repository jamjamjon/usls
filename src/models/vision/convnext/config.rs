use crate::NAMES_IMAGENET_1K;

///
/// > # ConvNeXt: A ConvNet for the 2020s
/// >
/// > Modern convolutional neural network that bridges the gap between CNNs and Vision Transformers.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
/// > - **Paper**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
/// >
/// > # Model Variants
/// >
/// > - **convnext-v1-tiny**: Tiny model for efficient inference
/// > - **convnext-v1-small**: Small model for balanced performance
/// > - **convnext-v1-base**: Base model for general use
/// > - **convnext-v1-large**: Large model for high accuracy
/// > - **convnext-v2-atto**: Ultra-small V2 model
/// > - **convnext-v2-femto**: Ultra-tiny V2 model
/// > - **convnext-v2-pico**: Tiny V2 model
/// > - **convnext-v2-nano**: Small V2 model
/// > - **convnext-v2-tiny**: V2 tiny model
/// > - **convnext-v2-small**: V2 small model
/// > - **convnext-v2-base**: V2 base model
/// > - **convnext-v2-large**: V2 large model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image Classification**: 1000-class ImageNet classification
/// >
/// Model configuration for `ConvNeXt`
///
impl crate::Config {
    /// Base configuration for ConvNeXt models
    pub fn convnext() -> Self {
        Self::default()
            .with_name("convnext")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 224)
            .with_model_ixx(0, 3, 224)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_apply_softmax(true)
            .with_topk(5)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    /// V1 tiny model for efficient inference
    pub fn convnext_v1_tiny() -> Self {
        Self::convnext().with_model_file("v1-t.onnx")
    }

    /// V1 small model for balanced performance
    pub fn convnext_v1_small() -> Self {
        Self::convnext().with_model_file("v1-s.onnx")
    }

    /// V1 base model for general use
    pub fn convnext_v1_base() -> Self {
        Self::convnext().with_model_file("v1-b.onnx")
    }

    /// V1 large model for high accuracy
    pub fn convnext_v1_large() -> Self {
        Self::convnext().with_model_file("v1-l.onnx")
    }

    /// V2 atto model (ultra-small)
    pub fn convnext_v2_atto() -> Self {
        Self::convnext().with_model_file("v2-a.onnx")
    }

    /// V2 femto model (ultra-tiny)
    pub fn convnext_v2_femto() -> Self {
        Self::convnext().with_model_file("v2-f.onnx")
    }

    /// V2 pico model (tiny)
    pub fn convnext_v2_pico() -> Self {
        Self::convnext().with_model_file("v2-p.onnx")
    }

    /// V2 nano model (small)
    pub fn convnext_v2_nano() -> Self {
        Self::convnext().with_model_file("v2-n.onnx")
    }

    /// V2 tiny model
    pub fn convnext_v2_tiny() -> Self {
        Self::convnext().with_model_file("v2-t.onnx")
    }

    /// V2 small model
    pub fn convnext_v2_small() -> Self {
        Self::convnext().with_model_file("v2-s.onnx")
    }

    /// V2 base model
    pub fn convnext_v2_base() -> Self {
        Self::convnext().with_model_file("v2-b.onnx")
    }

    /// V2 large model
    pub fn convnext_v2_large() -> Self {
        Self::convnext().with_model_file("v2-l.onnx")
    }
}
