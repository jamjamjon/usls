use crate::{
    models::YOLOPredsFormat, ModelConfig, ResizeMode, Scale, Task, NAMES_COCO_80,
    NAMES_COCO_KEYPOINTS_17, NAMES_IMAGENET_1K, NAMES_YOLO_DOCLAYOUT_10,
};

impl ModelConfig {
    pub fn yolo() -> Self {
        Self::default()
            .with_name("yolo")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
            .with_class_names(&NAMES_COCO_80)
    }

    pub fn yolo_classify() -> Self {
        Self::yolo()
            .with_task(Task::ImageClassification)
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_resize_mode(ResizeMode::FitExact)
            .with_resize_filter("Bilinear")
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    pub fn yolo_detect() -> Self {
        Self::yolo().with_task(Task::ObjectDetection)
    }

    pub fn yolo_pose() -> Self {
        Self::yolo()
            .with_task(Task::KeypointsDetection)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    pub fn yolo_segment() -> Self {
        Self::yolo().with_task(Task::InstanceSegmentation)
    }

    pub fn yolo_obb() -> Self {
        Self::yolo().with_task(Task::OrientedObjectDetection)
    }

    pub fn auto_yolo_model_file(mut self) -> Self {
        if self.model.file.is_empty() {
            // [version]-[scale]-[task]
            let mut y = String::new();
            if let Some(x) = self.version() {
                y.push_str(&x.to_string());
            }
            if let Some(x) = self.scale() {
                y.push_str(&format!("-{}", x));
            }
            if let Some(x) = self.task() {
                y.push_str(&format!("-{}", x.yolo_str()));
            }
            y.push_str(".onnx");
            self.model.file = y;
        }

        self
    }

    pub fn doclayout_yolo_docstructbench() -> Self {
        Self::yolo_detect()
            .with_version(10.into())
            .with_model_ixx(0, 2, (640, 1024, 1024).into())
            .with_model_ixx(0, 3, (640, 1024, 1024).into())
            .with_class_confs(&[0.4])
            .with_class_names(&NAMES_YOLO_DOCLAYOUT_10)
            .with_model_file("doclayout-docstructbench.onnx") // TODO: batch_size > 1
    }

    // YOLOE models
    pub fn yoloe_v8s_seg_pf() -> Self {
        Self::yolo_segment()
            .with_version(8.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-v8s-seg-pf.onnx")
    }

    pub fn yoloe_v8m_seg_pf() -> Self {
        Self::yolo_segment()
            .with_version(8.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-pf.onnx")
    }

    pub fn yoloe_v8l_seg_pf() -> Self {
        Self::yolo_segment()
            .with_version(8.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-v8l-seg-pf.onnx")
    }

    pub fn yoloe_11s_seg_pf() -> Self {
        Self::yolo_segment()
            .with_version(11.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-11s-seg-pf.onnx")
    }

    pub fn yoloe_11m_seg_pf() -> Self {
        Self::yolo_segment()
            .with_version(11.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-pf.onnx")
    }

    pub fn yoloe_11l_seg_pf() -> Self {
        Self::yolo_segment()
            .with_version(11.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-11l-seg-pf.onnx")
    }

    /// ---- TODO
    pub fn fastsam_s() -> Self {
        Self::yolo_segment()
            .with_scale(Scale::S)
            .with_version(8.into())
            .with_model_file("FastSAM-s.onnx")
    }

    pub fn yolo_v8_rtdetr_l() -> Self {
        Self::yolo_detect()
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
            .with_scale(Scale::L)
            .with_model_file("rtdetr-l-det.onnx")
    }
}
