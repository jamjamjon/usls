//! Convenience methods for inference params
impl crate::Config {
    /// Set class names from a slice of static strings.
    pub fn with_class_names(mut self, names: &[&'static str]) -> Self {
        self.inference.class_names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set class names from a Vec<String> (owned).
    pub fn with_class_names_owned(mut self, names: Vec<String>) -> Self {
        self.inference.class_names = names;
        self
    }

    /// Set class names (secondary) from a slice of static strings.
    pub fn with_class_names2(mut self, names: &[&'static str]) -> Self {
        self.inference.class_names2 = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set text names from a Vec<String> (owned).
    pub fn with_text_names_owned(mut self, names: Vec<String>) -> Self {
        self.inference.text_names = names;
        self
    }

    /// Set class confidence thresholds.
    pub fn with_class_confs(mut self, confs: &[f32]) -> Self {
        self.inference.class_confs = confs.to_vec();
        self
    }

    /// Set keypoint names from a slice of static strings.
    pub fn with_keypoint_names(mut self, names: &[&'static str]) -> Self {
        self.inference.keypoint_names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set keypoint names from a Vec<String> (owned).
    pub fn with_keypoint_names_owned(mut self, names: Vec<String>) -> Self {
        self.inference.keypoint_names = names;
        self
    }

    /// Set keypoint confidence thresholds.
    pub fn with_keypoint_confs(mut self, confs: &[f32]) -> Self {
        self.inference.keypoint_confs = confs.to_vec();
        self
    }

    /// Set text confidence thresholds.
    pub fn with_text_confs(mut self, confs: &[f32]) -> Self {
        self.inference.text_confs = confs.to_vec();
        self
    }

    /// Set IOU threshold.
    pub fn with_iou(mut self, iou: f32) -> Self {
        self.inference.iou = Some(iou);
        self
    }

    /// Set number of keypoints.
    pub fn with_nk(mut self, nk: usize) -> Self {
        self.inference.num_keypoints = Some(nk);
        self
    }

    /// Set minimum width.
    pub fn with_min_width(mut self, width: usize) -> Self {
        self.inference.min_width = Some(width as f32);
        self
    }

    /// Set minimum height.
    pub fn with_min_height(mut self, height: usize) -> Self {
        self.inference.min_height = Some(height as f32);
        self
    }

    /// Set apply softmax flag.
    pub fn with_apply_softmax(mut self, apply: bool) -> Self {
        self.inference.apply_softmax = Some(apply);
        self
    }

    /// Set topk value.
    pub fn with_topk(mut self, topk: usize) -> Self {
        self.inference.topk = Some(topk);
        self
    }

    /// Set DB binary threshold.
    pub fn with_db_binary_thresh(mut self, thresh: f32) -> Self {
        self.inference.db_binary_thresh = Some(thresh);
        self
    }

    /// Set DB unclip ratio.
    pub fn with_db_unclip_ratio(mut self, ratio: f32) -> Self {
        self.inference.db_unclip_ratio = Some(ratio);
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.inference.max_tokens = Some(max_tokens);
        self
    }

    /// Get class confidences (accessor for inference params).
    pub fn class_confs(&self) -> &[f32] {
        &self.inference.class_confs
    }

    /// Get text names accessor.
    pub fn text_names(&self) -> &[String] {
        &self.inference.text_names
    }

    /// Get token level class flag.
    pub fn token_level_class(&self) -> bool {
        self.inference.token_level_class
    }

    /// Set token level class flag.
    pub fn with_token_level_class(mut self, token_level_class: bool) -> Self {
        self.inference.token_level_class = token_level_class;
        self
    }

    /// Get text confidences.
    pub fn text_confs(&self) -> &[f32] {
        &self.inference.text_confs
    }

    /// Exclude specific classes from inference.
    pub fn exclude_classes(mut self, xs: &[usize]) -> Self {
        self.inference.classes_retained.clear();
        self.inference.classes_excluded.extend_from_slice(xs);
        self
    }

    /// Retain only specific classes in inference.
    pub fn retain_classes(mut self, xs: &[usize]) -> Self {
        self.inference.classes_excluded.clear();
        self.inference.classes_retained.extend_from_slice(xs);
        self
    }

    /// Set find contours flag.
    pub fn with_find_contours(mut self, find: bool) -> Self {
        self.inference.find_contours = find;
        self
    }

    /// Set number of classes.
    pub fn with_nc(mut self, nc: usize) -> Self {
        self.inference.num_classes = Some(nc);
        self
    }

    /// Set SAM kind.
    #[cfg(feature = "vision")]
    pub fn with_sam_kind(mut self, kind: crate::SamKind) -> Self {
        self.inference.sam_kind = Some(kind);
        self
    }

    /// Set SAM low resolution mask flag.
    #[cfg(feature = "vision")]
    pub fn with_sam_low_res_mask(mut self, low_res: bool) -> Self {
        self.inference.sam_low_res_mask = Some(low_res);
        self
    }

    /// Set YOLO predictions format.
    #[cfg(feature = "vision")]
    pub fn with_yolo_preds_format(mut self, format: crate::YOLOPredsFormat) -> Self {
        self.inference.yolo_preds_format = Some(format);
        self
    }
}
