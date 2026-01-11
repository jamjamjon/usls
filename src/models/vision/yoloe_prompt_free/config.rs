use crate::{Config, ResizeFilter, Scale, Task, Version, NAMES_YOLOE_4585};

///
/// > # YOLOE: Real-Time Seeing Anything
/// >
/// > Real-time object detection system with prompt-free capabilities for comprehensive vision tasks.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)
/// >
/// > # Model Variants
/// >
/// > - **yoloe-v8s-seg-pf**: YOLOE v8 small prompt-free segmentation
/// > - **yoloe-v8m-seg-pf**: YOLOE v8 medium prompt-free segmentation
/// > - **yoloe-v8l-seg-pf**: YOLOE v8 large prompt-free segmentation
/// > - **yoloe-11s-seg-pf**: YOLOE 11 small prompt-free segmentation
/// > - **yoloe-11m-seg-pf**: YOLOE 11 medium prompt-free segmentation
/// > - **yoloe-11l-seg-pf**: YOLOE 11 large prompt-free segmentation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Prompt-Free Segmentation**: Instance segmentation without prompts
/// > - [X] **Real-Time Performance**: Optimized for real-time inference
/// > - [X] **Multi-Scale Support**: Various model scales (S/M/L)
/// > - [X] **4585 Classes**: Large-scale object detection
/// >
/// Model configuration for `YOLOE Prompt Free`
///
impl Config {
    /// Base configuration for YOLOE models
    pub fn yoloe() -> Self {
        Self::default()
            .with_name("yoloe")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_resize_filter(ResizeFilter::CatmullRom)
            .with_task(Task::InstanceSegmentation)
    }

    /// Base configuration for YOLOE prompt-free segmentation
    pub fn yoloe_seg_pf() -> Self {
        Self::yolo()
            .with_class_names(&NAMES_YOLOE_4585)
            .with_task(Task::InstanceSegmentation)
    }

    /// YOLOE v8 small prompt-free segmentation
    pub fn yoloe_v8s_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(8))
            .with_scale(Scale::S)
            .with_model_file("yoloe-v8s-seg-pf.onnx")
    }

    /// YOLOE v8 medium prompt-free segmentation
    pub fn yoloe_v8m_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(8))
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-pf.onnx")
    }

    /// YOLOE v8 large prompt-free segmentation
    pub fn yoloe_v8l_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(8))
            .with_scale(Scale::L)
            .with_model_file("yoloe-v8l-seg-pf.onnx")
    }

    /// YOLOE 11 small prompt-free segmentation
    pub fn yoloe_11s_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(11))
            .with_scale(Scale::S)
            .with_model_file("yoloe-11s-seg-pf.onnx")
    }

    /// YOLOE 11 medium prompt-free segmentation
    pub fn yoloe_11m_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(11))
            .with_scale(Scale::M)
            .with_model_file("yoloe-11m-seg-pf.onnx")
    }

    /// YOLOE 11 large prompt-free segmentation
    pub fn yoloe_11l_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(11))
            .with_scale(Scale::L)
            .with_model_file("yoloe-11l-seg-pf.onnx")
    }
}
