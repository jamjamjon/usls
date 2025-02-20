use crate::{models::YOLOPredsFormat, Options, ResizeMode, Scale, Task, COCO_KEYPOINTS_NAMES_17};

impl Options {
    pub fn yolo() -> Self {
        Self::default()
            .with_model_name("yolo")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
            .with_find_contours(true)
    }

    pub fn doclayout_yolo_docstructbench() -> Self {
        Self::yolo_v10()
            .with_model_file("doclayout-docstructbench.onnx") // TODO: batch_size > 1
            .with_model_ixx(0, 2, (640, 1024, 1024).into())
            .with_model_ixx(0, 3, (640, 1024, 1024).into())
            .with_class_confs(&[0.4])
            .with_class_names(&[
                "title",
                "plain text",
                "abandon",
                "figure",
                "figure_caption",
                "table",
                "table_caption",
                "table_footnote",
                "isolate_formula",
                "formula_caption",
            ])
    }

    pub fn yolo_classify() -> Self {
        Self::yolo()
            .with_model_task(Task::ImageClassification)
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_resize_mode(ResizeMode::FitExact)
            .with_resize_filter("Bilinear")
    }

    pub fn yolo_detect() -> Self {
        Self::yolo().with_model_task(Task::ObjectDetection)
    }

    pub fn yolo_pose() -> Self {
        Self::yolo()
            .with_model_task(Task::KeypointsDetection)
            .with_keypoint_names(&COCO_KEYPOINTS_NAMES_17)
    }

    pub fn yolo_segment() -> Self {
        Self::yolo().with_model_task(Task::InstanceSegmentation)
    }

    pub fn yolo_obb() -> Self {
        Self::yolo().with_model_task(Task::OrientedObjectDetection)
    }

    pub fn fastsam_s() -> Self {
        Self::yolo_segment()
            .with_model_scale(Scale::S)
            .with_model_version(8.0.into())
            .with_model_file("FastSAM-s.onnx")
    }

    pub fn yolo_v8_rtdetr() -> Self {
        Self::yolo()
            .with_model_version(7.0.into())
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
    }

    pub fn yolo_v8_rtdetr_l() -> Self {
        Self::yolo_v8_rtdetr()
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
            .with_model_scale(Scale::L)
            .with_model_file("rtdetr-l-det.onnx")
    }

    pub fn yolo_v8_rtdetr_x() -> Self {
        Self::yolo_v8_rtdetr()
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
            .with_model_scale(Scale::X)
    }

    pub fn yolo_n() -> Self {
        Self::yolo().with_model_scale(Scale::N)
    }

    pub fn yolo_s() -> Self {
        Self::yolo().with_model_scale(Scale::S)
    }

    pub fn yolo_m() -> Self {
        Self::yolo().with_model_scale(Scale::M)
    }

    pub fn yolo_l() -> Self {
        Self::yolo().with_model_scale(Scale::L)
    }

    pub fn yolo_x() -> Self {
        Self::yolo().with_model_scale(Scale::X)
    }

    pub fn yolo_v5() -> Self {
        Self::yolo().with_model_version(5.0.into())
    }

    pub fn yolo_v6() -> Self {
        Self::yolo().with_model_version(6.0.into())
    }

    pub fn yolo_v7() -> Self {
        Self::yolo().with_model_version(7.0.into())
    }

    pub fn yolo_v8() -> Self {
        Self::yolo().with_model_version(8.0.into())
    }

    pub fn yolo_v9() -> Self {
        Self::yolo().with_model_version(9.0.into())
    }

    pub fn yolo_v10() -> Self {
        Self::yolo().with_model_version(10.0.into())
    }

    pub fn yolo_v11() -> Self {
        Self::yolo().with_model_version(11.0.into())
    }

    pub fn yolo_v12() -> Self {
        Self::yolo().with_model_version(12.0.into())
    }

    pub fn yolo_v8_n() -> Self {
        Self::yolo()
            .with_model_version(8.0.into())
            .with_model_scale(Scale::N)
    }

    pub fn yolo_v8_s() -> Self {
        Self::yolo()
            .with_model_version(8.0.into())
            .with_model_scale(Scale::S)
    }

    pub fn yolo_v8_m() -> Self {
        Self::yolo()
            .with_model_version(8.0.into())
            .with_model_scale(Scale::M)
    }

    pub fn yolo_v8_l() -> Self {
        Self::yolo()
            .with_model_version(8.0.into())
            .with_model_scale(Scale::L)
    }

    pub fn yolo_v8_x() -> Self {
        Self::yolo()
            .with_model_version(8.0.into())
            .with_model_scale(Scale::X)
    }

    pub fn yolo_v11_n() -> Self {
        Self::yolo()
            .with_model_version(11.0.into())
            .with_model_scale(Scale::N)
    }

    pub fn yolo_v11_s() -> Self {
        Self::yolo()
            .with_model_version(11.0.into())
            .with_model_scale(Scale::S)
    }

    pub fn yolo_v11_m() -> Self {
        Self::yolo()
            .with_model_version(11.0.into())
            .with_model_scale(Scale::M)
    }

    pub fn yolo_v11_l() -> Self {
        Self::yolo()
            .with_model_version(11.0.into())
            .with_model_scale(Scale::L)
    }

    pub fn yolo_v11_x() -> Self {
        Self::yolo()
            .with_model_version(11.0.into())
            .with_model_scale(Scale::X)
    }
}
