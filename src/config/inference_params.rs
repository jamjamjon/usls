use aksr::Builder;

/// Inference parameters for model execution.
///
/// This struct contains all runtime parameters that control inference behavior,
/// separate from model configuration (modules and processors).
#[derive(Builder, Debug, Clone, PartialEq)]
pub struct InferenceParams {
    // Detection parameters
    pub class_names: Vec<String>,
    pub class_names2: Vec<String>,
    pub class_confs: Vec<f32>,
    pub num_classes: Option<usize>,
    pub iou: Option<f32>,
    pub apply_nms: Option<bool>,
    pub topk: Option<usize>,
    pub classes_excluded: Vec<usize>,
    pub classes_retained: Vec<usize>,
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,

    // Keypoint parameters
    pub keypoint_names: Vec<String>,
    pub keypoint_confs: Vec<f32>,
    pub num_keypoints: Option<usize>,

    // Segmentation parameters
    pub num_masks: Option<usize>,
    pub find_contours: bool,

    // OCR/Text parameters
    pub text_names: Vec<String>,
    pub text_confs: Vec<f32>,
    pub db_unclip_ratio: Option<f32>,
    pub db_binary_thresh: Option<f32>,
    pub token_level_class: bool,

    // Task-specific parameters
    #[cfg(feature = "vision")]
    pub yolo_preds_format: Option<crate::YOLOPredsFormat>,
    #[cfg(feature = "vision")]
    pub sam_kind: Option<crate::SamKind>,
    #[cfg(feature = "vision")]
    pub sam_low_res_mask: Option<bool>,

    /// Super Resolution: up-scaling factor.
    pub up_scale: f32,

    // Common parameters
    pub apply_softmax: Option<bool>,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            class_confs: vec![0.25],
            keypoint_confs: vec![0.35],
            text_confs: vec![0.25],
            apply_softmax: Some(false),
            db_unclip_ratio: Some(1.5),
            db_binary_thresh: Some(0.2),
            class_names: Default::default(),
            class_names2: Default::default(),
            num_classes: Default::default(),
            iou: Default::default(),
            apply_nms: Default::default(),
            topk: Default::default(),
            classes_excluded: Default::default(),
            classes_retained: Default::default(),
            min_width: Default::default(),
            min_height: Default::default(),
            keypoint_names: Default::default(),
            num_keypoints: Default::default(),
            num_masks: Default::default(),
            find_contours: Default::default(),
            up_scale: 2.0,
            text_names: Default::default(),
            token_level_class: Default::default(),
            #[cfg(feature = "vision")]
            yolo_preds_format: Default::default(),
            #[cfg(feature = "vision")]
            sam_kind: Default::default(),
            #[cfg(feature = "vision")]
            sam_low_res_mask: Default::default(),
        }
    }
}
