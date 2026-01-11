///
/// > # YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception
/// >
/// > Multi-task model for autonomous driving with object detection, drivable area segmentation, and lane detection.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)
/// >
/// > # Model Variants
/// >
/// > - **yolop-v2-480x800**: YOLOPv2 model for 480x800 resolution
/// > - **yolop-v2-736x1280**: YOLOPv2 model for 736x1280 resolution
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Object Detection**: Traffic object detection
/// > - [X] **Drivable Area Segmentation**: Road area segmentation
/// > - [X] **Lane Detection**: Lane marking detection
/// > - [X] **Multi-Resolution**: Support for different input resolutions
/// >
/// Model configuration for `YOLOP`
///
impl crate::Config {
    /// Base configuration for YOLOP models
    pub fn yolop() -> Self {
        Self::default()
            .with_name("yolop")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_class_confs(&[0.3])
    }

    /// YOLOPv2 model for 480x800 resolution
    pub fn yolop_v2_480x800() -> Self {
        Self::yolop().with_model_file("v2-480x800.onnx")
    }

    /// YOLOPv2 model for 736x1280 resolution
    pub fn yolop_v2_736x1280() -> Self {
        Self::yolop().with_model_file("v2-736x1280.onnx")
    }
}
