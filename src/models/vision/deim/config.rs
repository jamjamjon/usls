/// Model configuration for `DEIM`
impl crate::Config {
    pub fn deim() -> Self {
        Self::d_fine().with_name("deim")
    }

    pub fn deim_dfine_s_coco() -> Self {
        Self::deim().with_model_file("dfine-s-coco.onnx")
    }

    pub fn deim_dfine_m_coco() -> Self {
        Self::deim().with_model_file("dfine-m-coco.onnx")
    }

    pub fn deim_dfine_l_coco() -> Self {
        Self::deim().with_model_file("dfine-l-coco.onnx")
    }

    pub fn deim_dfine_x_coco() -> Self {
        Self::deim().with_model_file("dfine-x-coco.onnx")
    }
}
