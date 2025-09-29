/// Model configuration for `DEIMv2`
impl crate::Config {
    pub fn deimv2() -> Self {
        Self::d_fine().with_name("deimv2")
    }

    pub fn deim_v2_atto_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-atto-coco.onnx")
    }

    pub fn deim_v2_femto_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-femto-coco.onnx")
    }

    pub fn deim_v2_pico_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-pico-coco.onnx")
    }

    pub fn deim_v2_n_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-n-coco.onnx")
    }

    pub fn deim_v2_s_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-s-coco.onnx")
    }

    pub fn deim_v2_m_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-m-coco.onnx")
    }

    pub fn deim_v2_l_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-l-coco.onnx")
    }

    pub fn deim_v2_x_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-x-coco.onnx")
    }
}
