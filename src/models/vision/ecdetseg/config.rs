use crate::Task;

const ECDETSEG_RELEASE: &str = "https://github.com/wep21/assets/releases/download/ecdetseg";

///
/// > # ECDetSeg: EdgeCrafter Instance Segmentation
/// >
/// > Compact ViT-based real-time instance segmentation from EdgeCrafter, built on the
/// > RT-DETR / D-FINE detection family with an additional mask head.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)
/// > - **arXiv**: <https://arxiv.org/abs/2603.18739>
/// >
/// > # Model Variants
/// >
/// > - **ecdetseg-s**: Small model for 80-class COCO instance segmentation
/// > - **ecdetseg-m**: Medium model for 80-class COCO instance segmentation
/// > - **ecdetseg-l**: Large model for 80-class COCO instance segmentation
/// > - **ecdetseg-x**: Extra large model for 80-class COCO instance segmentation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Instance Segmentation**: 80-class COCO instance segmentation
/// >
/// > # Precision / File naming
/// >
/// > Assets are hosted on the `wep21/assets` GitHub releases (tag `ecdetseg`).
/// > FP32 weights use `ecdetseg-{scale}.onnx`; FP16 weights follow the `-fp16.onnx`
/// > convention (`ecdetseg-{scale}-fp16.onnx`) and are selected automatically via
/// > [`crate::Config::with_model_dtype(DType::Fp16)`](crate::Config::with_model_dtype).
///
/// Model configuration for `ECDetSeg`
///
impl crate::Config {
    /// Base configuration for ECDetSeg models.
    ///
    /// ECDetSeg shares the RT-DETR / D-FINE detection front-end (same `images` +
    /// `orig_target_sizes` inputs and `labels`, `boxes`, `scores` outputs) and adds a
    /// fourth `masks` output for instance segmentation.
    pub fn ecdetseg() -> Self {
        Self::rtdetr()
            .with_name("ecdetseg")
            .with_task(Task::InstanceSegmentation)
            // EdgeCrafter eval stretches to the model size (letterbox degrades small models)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
    }

    /// Small model for 80-class COCO instance segmentation.
    pub fn ecdetseg_s() -> Self {
        Self::ecdetseg().with_model_file(format!("{ECDETSEG_RELEASE}/ecdetseg-s.onnx"))
    }

    /// Medium model for 80-class COCO instance segmentation.
    pub fn ecdetseg_m() -> Self {
        Self::ecdetseg().with_model_file(format!("{ECDETSEG_RELEASE}/ecdetseg-m.onnx"))
    }

    /// Large model for 80-class COCO instance segmentation.
    pub fn ecdetseg_l() -> Self {
        Self::ecdetseg().with_model_file(format!("{ECDETSEG_RELEASE}/ecdetseg-l.onnx"))
    }

    /// Extra large model for 80-class COCO instance segmentation.
    pub fn ecdetseg_x() -> Self {
        Self::ecdetseg().with_model_file(format!("{ECDETSEG_RELEASE}/ecdetseg-x.onnx"))
    }
}
