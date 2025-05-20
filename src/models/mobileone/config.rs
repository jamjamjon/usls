use crate::NAMES_IMAGENET_1K;

/// Model configuration for `MobileOne`
impl crate::Config {
    pub fn mobileone() -> Self {
        Self::default()
            .with_name("mobileone")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_apply_softmax(true)
            .with_normalize(true)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    pub fn mobileone_s0() -> Self {
        Self::mobileone().with_model_file("s0.onnx")
    }

    pub fn mobileone_s1() -> Self {
        Self::mobileone().with_model_file("s1.onnx")
    }

    pub fn mobileone_s2() -> Self {
        Self::mobileone().with_model_file("s2.onnx")
    }

    pub fn mobileone_s3() -> Self {
        Self::mobileone().with_model_file("s3.onnx")
    }

    pub fn mobileone_s4_224x224() -> Self {
        Self::mobileone().with_model_file("s4-224x224.onnx")
    }

    pub fn mobileone_s4_256x256() -> Self {
        Self::mobileone().with_model_file("s4-256x256.onnx")
    }

    pub fn mobileone_s4_384x384() -> Self {
        Self::mobileone().with_model_file("s4-384x384.onnx")
    }

    pub fn mobileone_s4_512x512() -> Self {
        Self::mobileone().with_model_file("s4-512x512.onnx")
    }
}
