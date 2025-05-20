use crate::NAMES_IMAGENET_1K;

/// Model configuration for `ConvNeXt`
impl crate::Config {
    pub fn convnext() -> Self {
        Self::default()
            .with_name("convnext")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_apply_softmax(true)
            .with_topk(5)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    pub fn convnext_v1_tiny() -> Self {
        Self::convnext().with_model_file("v1-t.onnx")
    }

    pub fn convnext_v1_small() -> Self {
        Self::convnext().with_model_file("v1-s.onnx")
    }

    pub fn convnext_v1_base() -> Self {
        Self::convnext().with_model_file("v1-b.onnx")
    }

    pub fn convnext_v1_large() -> Self {
        Self::convnext().with_model_file("v1-l.onnx")
    }

    pub fn convnext_v2_atto() -> Self {
        Self::convnext().with_model_file("v2-a.onnx")
    }

    pub fn convnext_v2_femto() -> Self {
        Self::convnext().with_model_file("v2-f.onnx")
    }

    pub fn convnext_v2_pico() -> Self {
        Self::convnext().with_model_file("v2-p.onnx")
    }

    pub fn convnext_v2_nano() -> Self {
        Self::convnext().with_model_file("v2-n.onnx")
    }

    pub fn convnext_v2_tiny() -> Self {
        Self::convnext().with_model_file("v2-t.onnx")
    }

    pub fn convnext_v2_small() -> Self {
        Self::convnext().with_model_file("v2-s.onnx")
    }

    pub fn convnext_v2_base() -> Self {
        Self::convnext().with_model_file("v2-b.onnx")
    }

    pub fn convnext_v2_large() -> Self {
        Self::convnext().with_model_file("v2-l.onnx")
    }
}
