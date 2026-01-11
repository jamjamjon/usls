use crate::{Config, Scale, Task};

///
/// > # FastSAM: Fast Segment Anything
/// >
/// > Real-time image segmentation model with CNN-based architecture for fast inference.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [CASIA-LMC-Lab/FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM)
/// > - **Paper**: [Fast Segment Anything](https://arxiv.org/abs/2306.12156)
/// >
/// > # Model Variants
/// >
/// > - **fastsam-s**: Small model for real-time segmentation
/// > - **fastsam-x**: Extra large model for high accuracy
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Instance Segmentation**: Real-time object segmentation
/// > - [X] **Fast Inference**: CNN-based architecture for speed
/// >
/// Model configuration for `FastSAM`
///
impl Config {
    /// Base configuration for FastSAM models
    pub fn fastsam() -> Self {
        Self::default()
            .with_name("fastsam")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_task(Task::InstanceSegmentation)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_resize_filter(crate::ResizeFilter::CatmullRom)
            .with_version(8.into())
            .with_class_names(&["object"])
    }

    /// Small model for real-time segmentation
    pub fn fastsam_s() -> Self {
        Self::fastsam()
            .with_scale(Scale::S)
            .with_model_file("FastSAM-s.onnx")
    }

    /// Extra large model for high accuracy
    pub fn fastsam_x() -> Self {
        Self::fastsam()
            .with_scale(Scale::X)
            .with_model_file("FastSAM-x.onnx")
    }
}
