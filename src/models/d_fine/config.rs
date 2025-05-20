/// Model configuration for `d_fine`
impl crate::Config {
    pub fn d_fine() -> Self {
        Self::rtdetr().with_name("d-fine")
    }

    pub fn d_fine_n_coco() -> Self {
        Self::d_fine().with_model_file("n-coco.onnx")
    }

    pub fn d_fine_s_coco() -> Self {
        Self::d_fine().with_model_file("s-coco.onnx")
    }

    pub fn d_fine_m_coco() -> Self {
        Self::d_fine().with_model_file("m-coco.onnx")
    }

    pub fn d_fine_l_coco() -> Self {
        Self::d_fine().with_model_file("l-coco.onnx")
    }

    pub fn d_fine_x_coco() -> Self {
        Self::d_fine().with_model_file("x-coco.onnx")
    }

    pub fn d_fine_s_coco_obj365() -> Self {
        Self::d_fine().with_model_file("s-obj2coco.onnx")
    }

    pub fn d_fine_m_coco_obj365() -> Self {
        Self::d_fine().with_model_file("m-obj2coco.onnx")
    }

    pub fn d_fine_l_coco_obj365() -> Self {
        Self::d_fine().with_model_file("l-obj2coco.onnx")
    }

    pub fn d_fine_x_coco_obj365() -> Self {
        Self::d_fine().with_model_file("x-obj2coco.onnx")
    }
}
