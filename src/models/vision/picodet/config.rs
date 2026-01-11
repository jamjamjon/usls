use crate::{
    NAMES_COCO_80, NAMES_PICODET_LAYOUT_17, NAMES_PICODET_LAYOUT_3, NAMES_PICODET_LAYOUT_5,
};

///
/// > # PP-PicoDet: A Better Real-Time Object Detector
/// >
/// > PaddlePaddle's real-time object detector optimized for mobile devices.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/picodet)
/// >
/// > # Model Variants
/// >
/// > - **picodet-l-coco**: Large model for COCO object detection
/// > - **picodet-layout-1x**: Layout analysis model (5 classes)
/// > - **picodet-l-layout-3cls**: Large layout model (3 classes)
/// > - **picodet-l-layout-17cls**: Large layout model (17 classes)
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Object Detection**: 80-class COCO object detection
/// > - [X] **Layout Analysis**: Document layout detection
/// > - [X] **Mobile Optimization**: Efficient inference on mobile devices
/// >
/// Model configuration for `PicoDet`
///
impl crate::Config {
    /// Base configuration for PicoDet models
    pub fn picodet() -> Self {
        Self::default()
            .with_name("picodet")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_model_ixx(1, 0, 1)
            .with_model_ixx(1, 1, 2)
            .with_batch_size_all(1)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_class_confs(&[0.5])
    }

    /// Large model for COCO object detection
    pub fn picodet_l_coco() -> Self {
        Self::picodet()
            .with_model_file("l-coco.onnx")
            .with_class_names(&NAMES_COCO_80)
    }

    /// Layout analysis model (5 classes)
    pub fn picodet_layout_1x() -> Self {
        Self::picodet()
            .with_model_file("layout-1x.onnx")
            .with_class_names(&NAMES_PICODET_LAYOUT_5)
    }

    /// Large layout model (3 classes)
    pub fn picodet_l_layout_3cls() -> Self {
        Self::picodet()
            .with_model_file("l-layout-3cls.onnx")
            .with_class_names(&NAMES_PICODET_LAYOUT_3)
    }

    /// Large layout model (17 classes)
    pub fn picodet_l_layout_17cls() -> Self {
        Self::picodet()
            .with_model_file("l-layout-17cls.onnx")
            .with_class_names(&NAMES_PICODET_LAYOUT_17)
    }
}
