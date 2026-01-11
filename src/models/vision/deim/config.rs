///
/// > # DEIM: DETR with Improved Matching
/// >
/// > DETR with improved matching for fast convergence and better performance.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [ShihuaHuang95/DEIM](https://github.com/ShihuaHuang95/DEIM)
/// >
/// > # Model Variants
/// >
/// > - **deim-dfine-s**: Small model based on D-FINE
/// > - **deim-dfine-m**: Medium model based on D-FINE
/// > - **deim-dfine-l**: Large model based on D-FINE
/// > - **deim-dfine-x**: Extra large model based on D-FINE
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Object Detection**: 80-class COCO object detection
/// >
/// Model configuration for `DEIM`
///
impl crate::Config {
    /// Base configuration for DEIM models
    pub fn deim() -> Self {
        Self::d_fine().with_name("deim")
    }

    /// Small model based on D-FINE
    pub fn deim_dfine_s_coco() -> Self {
        Self::deim().with_model_file("dfine-s-coco.onnx")
    }

    /// Medium model based on D-FINE
    pub fn deim_dfine_m_coco() -> Self {
        Self::deim().with_model_file("dfine-m-coco.onnx")
    }

    /// Large model based on D-FINE
    pub fn deim_dfine_l_coco() -> Self {
        Self::deim().with_model_file("dfine-l-coco.onnx")
    }

    /// Extra large model based on D-FINE
    pub fn deim_dfine_x_coco() -> Self {
        Self::deim().with_model_file("dfine-x-coco.onnx")
    }
}
