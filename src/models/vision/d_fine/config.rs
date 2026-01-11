///
/// > # D-FINE: Redefine Regression Task of DETRs
/// >
/// > DETR with fine-grained distribution refinement for real-time object detection.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [manhbd-22022602/D-FINE](https://github.com/manhbd-22022602/D-FINE)
/// >
/// > # Model Variants
/// >
/// > - **d-fine-n**: Nano model for mobile deployment
/// > - **d-fine-s**: Small model for edge devices
/// > - **d-fine-m**: Medium model for balanced performance
/// > - **d-fine-l**: Large model for high accuracy
/// > - **d-fine-x**: Extra large model for maximum accuracy
/// > - **d-fine-s-obj365**: Small model trained on COCO+Objects365
/// > - **d-fine-m-obj365**: Medium model trained on COCO+Objects365
/// > - **d-fine-l-obj365**: Large model trained on COCO+Objects365
/// > - **d-fine-x-obj365**: Extra large model trained on COCO+Objects365
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Object Detection**: 80-class COCO object detection
/// > - [X] **Object Detection + Obj365**: Combined COCO+Objects365 detection
/// >
/// Model configuration for `D-FINE`
///
impl crate::Config {
    /// Base configuration for D-FINE models
    pub fn d_fine() -> Self {
        Self::rtdetr().with_name("d-fine")
    }

    /// Nano model for mobile deployment
    pub fn d_fine_n_coco() -> Self {
        Self::d_fine().with_model_file("n-coco.onnx")
    }

    /// Small model for edge devices
    pub fn d_fine_s_coco() -> Self {
        Self::d_fine().with_model_file("s-coco.onnx")
    }

    /// Medium model for balanced performance
    pub fn d_fine_m_coco() -> Self {
        Self::d_fine().with_model_file("m-coco.onnx")
    }

    /// Large model for high accuracy
    pub fn d_fine_l_coco() -> Self {
        Self::d_fine().with_model_file("l-coco.onnx")
    }

    /// Extra large model for maximum accuracy
    pub fn d_fine_x_coco() -> Self {
        Self::d_fine().with_model_file("x-coco.onnx")
    }

    /// Small model trained on COCO+Objects365
    pub fn d_fine_s_coco_obj365() -> Self {
        Self::d_fine().with_model_file("s-obj2coco.onnx")
    }

    /// Medium model trained on COCO+Objects365
    pub fn d_fine_m_coco_obj365() -> Self {
        Self::d_fine().with_model_file("m-obj2coco.onnx")
    }

    /// Large model trained on COCO+Objects365
    pub fn d_fine_l_coco_obj365() -> Self {
        Self::d_fine().with_model_file("l-obj2coco.onnx")
    }

    /// Extra large model trained on COCO+Objects365
    pub fn d_fine_x_coco_obj365() -> Self {
        Self::d_fine().with_model_file("x-obj2coco.onnx")
    }
}
