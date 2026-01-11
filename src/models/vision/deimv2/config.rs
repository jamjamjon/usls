///
/// > # DEIMv2: Real-Time Object Detection Meets DINOv3
/// >
/// > Advanced real-time object detection combining DEIM with DINOv3 features.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
/// >
/// > # Model Variants
/// >
/// > - **deim-v2-atto**: Ultra-small model with HGNetV2 backbone
/// > - **deim-v2-femto**: Ultra-tiny model with HGNetV2 backbone
/// > - **deim-v2-pico**: Tiny model with HGNetV2 backbone
/// > - **deim-v2-n**: Small model with HGNetV2 backbone
/// > - **deim-v2-s**: Small model with DINOv3 backbone
/// > - **deim-v2-m**: Medium model with DINOv3 backbone
/// > - **deim-v2-l**: Large model with DINOv3 backbone
/// > - **deim-v2-x**: Extra large model with DINOv3 backbone
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Real-Time Object Detection**: 80-class COCO object detection
/// > - [X] **DINOv3 Integration**: Advanced vision transformer features
/// >
/// Model configuration for `DEIMv2`
///
impl crate::Config {
    /// Base configuration for DEIMv2 models
    pub fn deimv2() -> Self {
        Self::d_fine().with_name("deimv2")
    }

    /// Ultra-small model with HGNetV2 backbone
    pub fn deim_v2_atto_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-atto-coco.onnx")
    }

    /// Ultra-tiny model with HGNetV2 backbone
    pub fn deim_v2_femto_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-femto-coco.onnx")
    }

    /// Tiny model with HGNetV2 backbone
    pub fn deim_v2_pico_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-pico-coco.onnx")
    }

    /// Small model with HGNetV2 backbone
    pub fn deim_v2_n_coco() -> Self {
        Self::deimv2().with_model_file("hgnetv2-n-coco.onnx")
    }

    /// Small model with DINOv3 backbone
    pub fn deim_v2_s_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-s-coco.onnx")
    }

    /// Medium model with DINOv3 backbone
    pub fn deim_v2_m_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-m-coco.onnx")
    }

    /// Large model with DINOv3 backbone
    pub fn deim_v2_l_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-l-coco.onnx")
    }

    /// Extra large model with DINOv3 backbone
    pub fn deim_v2_x_coco() -> Self {
        Self::deimv2().with_model_file("dinov3-x-coco.onnx")
    }
}
