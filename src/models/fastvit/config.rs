use crate::NAMES_IMAGENET_1K;

/// Model configuration for `FastViT`
impl crate::Config {
    pub fn fastvit() -> Self {
        Self::default()
            .with_name("fastvit")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_apply_softmax(true)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    pub fn fastvit_t8() -> Self {
        Self::fastvit().with_model_file("t8.onnx")
    }

    pub fn fastvit_t8_distill() -> Self {
        Self::fastvit().with_model_file("t8-distill.onnx")
    }

    pub fn fastvit_t12() -> Self {
        Self::fastvit().with_model_file("t12.onnx")
    }

    pub fn fastvit_t12_distill() -> Self {
        Self::fastvit().with_model_file("t12-distill.onnx")
    }

    pub fn fastvit_s12() -> Self {
        Self::fastvit().with_model_file("s12.onnx")
    }

    pub fn fastvit_s12_distill() -> Self {
        Self::fastvit().with_model_file("s12-distill.onnx")
    }

    pub fn fastvit_sa12() -> Self {
        Self::fastvit().with_model_file("sa12.onnx")
    }

    pub fn fastvit_sa12_distill() -> Self {
        Self::fastvit().with_model_file("sa12-distill.onnx")
    }

    pub fn fastvit_sa24() -> Self {
        Self::fastvit().with_model_file("sa24.onnx")
    }

    pub fn fastvit_sa24_distill() -> Self {
        Self::fastvit().with_model_file("sa24-distill.onnx")
    }

    pub fn fastvit_sa36() -> Self {
        Self::fastvit().with_model_file("sa36.onnx")
    }

    pub fn fastvit_sa36_distill() -> Self {
        Self::fastvit().with_model_file("sa36-distill.onnx")
    }

    pub fn fastvit_ma36() -> Self {
        Self::fastvit().with_model_file("ma36.onnx")
    }

    pub fn fastvit_ma36_distill() -> Self {
        Self::fastvit().with_model_file("ma36-distill.onnx")
    }
}
