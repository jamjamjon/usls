use crate::NAMES_COCO_80;

///
/// > # RT-DETR: Real-Time Detection Transformer
/// >
/// > Real-time object detection transformer that beats YOLOs on performance.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub v1**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
/// > - **GitHub v2**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
/// > - **GitHub v4**: [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)
/// >
/// > # Model Variants
/// >
/// > - **rtdetr-v1-r18**: V1 ResNet-18 backbone
/// > - **rtdetr-v1-r34**: V1 ResNet-34 backbone
/// > - **rtdetr-v1-r50**: V1 ResNet-50 backbone
/// > - **rtdetr-v1-r101**: V1 ResNet-101 backbone
/// > - **rtdetr-v1-r18-obj365**: V1 ResNet-18 trained on Objects365
/// > - **rtdetr-v1-r50-obj365**: V1 ResNet-50 trained on Objects365
/// > - **rtdetr-v2-s**: V2 small model
/// > - **rtdetr-v2-ms**: V2 medium-small model
/// > - **rtdetr-v2-m**: V2 medium model
/// > - **rtdetr-v2-l**: V2 large model
/// > - **rtdetr-v2-x**: V2 extra large model
/// > - **rtdetr-v4-s**: V4 small model with vision foundation
/// > - **rtdetr-v4-m**: V4 medium model with vision foundation
/// > - **rtdetr-v4-l**: V4 large model with vision foundation
/// > - **rtdetr-v4-x**: V4 extra large model with vision foundation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Real-time Object Detection**: 80-class COCO object detection
/// > - [X] **Objects365 Detection**: Enhanced detection with Objects365 dataset
/// > - [X] **Vision Foundation Models**: V4 with advanced vision features
/// > - [X] **Multiple Backbones**: ResNet series support
/// >
/// Model configuration for `RT-DETR`
///
impl crate::Config {
    /// Base configuration for RT-DETR models
    pub fn rtdetr() -> Self {
        Self::default()
            .with_name("rtdetr")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_class_confs(&[0.5])
            .with_class_names(&NAMES_COCO_80)
    }

    /// V1 ResNet-18 backbone
    pub fn rtdetr_v1_r18() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r18.onnx")
    }

    /// V1 ResNet-18 trained on Objects365
    pub fn rtdetr_v1_r18_obj365() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r18-obj365.onnx")
    }

    /// V1 ResNet-34 backbone
    pub fn rtdetr_v1_r34() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r34.onnx")
    }

    /// V1 ResNet-50 backbone
    pub fn rtdetr_v1_r50() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r50.onnx")
    }

    /// V1 ResNet-50 trained on Objects365
    pub fn rtdetr_v1_r50_obj365() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r50-obj365.onnx")
    }

    /// V1 ResNet-101 backbone
    pub fn rtdetr_v1_r101() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r101.onnx")
    }

    /// V1 ResNet-101 trained on Objects365
    pub fn rtdetr_v1_r101_obj365() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r101-obj365.onnx")
    }

    /// V2 small model
    pub fn rtdetr_v2_s() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-s.onnx")
    }

    /// V2 medium-small model
    pub fn rtdetr_v2_ms() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-ms.onnx")
    }

    /// V2 medium model
    pub fn rtdetr_v2_m() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-m.onnx")
    }

    /// V2 large model
    pub fn rtdetr_v2_l() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-l.onnx")
    }

    /// V2 extra large model
    pub fn rtdetr_v2_x() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-x.onnx")
    }

    /// V4 small model with vision foundation
    pub fn rtdetr_v4_s() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-s.onnx")
    }

    /// V4 medium model with vision foundation
    pub fn rtdetr_v4_m() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-m.onnx")
    }

    /// V4 large model with vision foundation
    pub fn rtdetr_v4_l() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-l.onnx")
    }

    /// V4 extra large model with vision foundation
    pub fn rtdetr_v4_x() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-x.onnx")
    }
}
