use crate::{Config, Scale, Task, Version, NAMES_YOLOE_4585};

///
/// > # YOLOE: Real-Time Seeing Anything
/// >
/// > Real-time object detection system with prompt-free capabilities for comprehensive vision tasks.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)
/// > - **GitHub**: [ultralytics/YOLOE-26](https://github.com/ultralytics/ultralytics)
/// >
/// > # Model Variants
/// >
/// > - **yoloe-v8s-seg-pf**: YOLOE v8 small prompt-free segmentation
/// > - **yoloe-v8m-seg-pf**: YOLOE v8 medium prompt-free segmentation
/// > - **yoloe-v8l-seg-pf**: YOLOE v8 large prompt-free segmentation
/// > - **yoloe-11s-seg-pf**: YOLOE 11 small prompt-free segmentation
/// > - **yoloe-11m-seg-pf**: YOLOE 11 medium prompt-free segmentation
/// > - **yoloe-11l-seg-pf**: YOLOE 11 large prompt-free segmentation
/// > - **yoloe-26n-seg-pf**: YOLOE 26 small prompt-free segmentation
/// > - **yoloe-26s-seg-pf**: YOLOE 26 medium prompt-free segmentation
/// > - **yoloe-26m-seg-pf**: YOLOE 26 large prompt-free segmentation
/// > - **yoloe-26l-seg-pf**: YOLOE 26 extra large prompt-free segmentation
/// > - **yoloe-26x-seg-pf**: YOLOE 26 extra extra large prompt-free segmentation
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
        Self::yolo()
            .with_name("yoloe")
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

    /// YOLOE 26 nano prompt-free segmentation
    pub fn yoloe_26n_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(26))
            .with_scale(Scale::N)
            .with_model_file("yoloe-26n-seg-pf-one2one.onnx")
    }

    /// YOLOE 26 small prompt-free segmentation
    pub fn yoloe_26s_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(26))
            .with_scale(Scale::S)
            .with_model_file("yoloe-26s-seg-pf-one2one.onnx")
    }

    /// YOLOE 26 medium prompt-free segmentation
    pub fn yoloe_26m_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(26))
            .with_scale(Scale::M)
            .with_model_file("yoloe-26m-seg-pf-one2one.onnx")
    }

    /// YOLOE 26 large prompt-free segmentation
    pub fn yoloe_26l_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(26))
            .with_scale(Scale::L)
            .with_model_file("yoloe-26l-seg-pf-one2one.onnx")
    }

    /// YOLOE 26 extra large prompt-free segmentation
    pub fn yoloe_26x_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(Version::from(26))
            .with_scale(Scale::X)
            .with_model_file("yoloe-26x-seg-pf-one2one.onnx")
    }
}
