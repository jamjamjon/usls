use crate::NAMES_IMAGENET_1K;

///
/// > # MobileOne: An Improved One millisecond Mobile Backbone
/// >
/// > Ultra-fast mobile backbone achieving 1ms inference time with high accuracy.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [apple/ml-mobileone](https://github.com/apple/ml-mobileone)
/// > - **Paper**: [MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2306.02645)
/// >
/// > # Model Variants
/// >
/// > - **mobileone-s0**: Small model with minimal parameters
/// > - **mobileone-s1**: Small model with slightly more parameters
/// > - **mobileone-s2**: Small model with balanced performance
/// > - **mobileone-s3**: Small model with enhanced performance
/// > - **mobileone-s4-224x224**: Small model at 224x224 resolution
/// > - **mobileone-s4-256x256**: Small model at 256x256 resolution
/// > - **mobileone-s4-384x384**: Small model at 384x384 resolution
/// > - **mobileone-s4-512x512**: Small model at 512x512 resolution
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image Classification**: 1000-class ImageNet classification
/// > - [X] **Ultra-fast Inference**: 1ms inference time on mobile devices
/// > - [X] **Multiple Resolutions**: Support for various input sizes
/// >
/// Model configuration for `MobileOne`
///
impl crate::Config {
    /// Base configuration for MobileOne models
    pub fn mobileone() -> Self {
        Self::default()
            .with_name("mobileone")
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

    /// Small model with minimal parameters
    pub fn mobileone_s0() -> Self {
        Self::mobileone().with_model_file("s0.onnx")
    }

    /// Small model with slightly more parameters
    pub fn mobileone_s1() -> Self {
        Self::mobileone().with_model_file("s1.onnx")
    }

    /// Small model with balanced performance
    pub fn mobileone_s2() -> Self {
        Self::mobileone().with_model_file("s2.onnx")
    }

    /// Small model with enhanced performance
    pub fn mobileone_s3() -> Self {
        Self::mobileone().with_model_file("s3.onnx")
    }

    /// Small model at 224x224 resolution
    pub fn mobileone_s4_224x224() -> Self {
        Self::mobileone().with_model_file("s4-224x224.onnx")
    }

    /// Small model at 256x256 resolution
    pub fn mobileone_s4_256x256() -> Self {
        Self::mobileone().with_model_file("s4-256x256.onnx")
    }

    /// Small model at 384x384 resolution
    pub fn mobileone_s4_384x384() -> Self {
        Self::mobileone().with_model_file("s4-384x384.onnx")
    }

    /// Small model at 512x512 resolution
    pub fn mobileone_s4_512x512() -> Self {
        Self::mobileone().with_model_file("s4-512x512.onnx")
    }
}
