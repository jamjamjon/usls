use crate::{Task, TensorView};

/// Bounding box coordinate format types.
#[derive(Debug, Clone, PartialEq)]
pub enum BoxType {
    Cxcywh,
    Cxcyxy,
    Xyxy,
    Xywh,
    XyCxcy,
}

/// Classification output format types.
#[derive(Debug, Clone, PartialEq)]
pub enum ClssType {
    Clss,
    ConfCls,
    ClsConf,
    ConfClss,
    ClssConf,
}

/// Keypoint output format types.
#[derive(Debug, Clone, PartialEq)]
pub enum KptsType {
    Xys,
    Xycs,
}

/// Anchor position in the prediction pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum AnchorsPosition {
    Before,
    After,
}

/// YOLO prediction format configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct YOLOPredsFormat {
    pub clss: ClssType,
    pub bbox: Option<BoxType>,
    pub kpts: Option<KptsType>,
    pub coefs: Option<bool>,
    pub obb: Option<bool>,
    pub anchors: Option<AnchorsPosition>,
    pub is_bbox_normalized: bool,
    pub apply_nms: bool,
    pub apply_softmax: bool,
    // ------------------------------------------------
    // pub is_concatenated: bool, // TODO: how to tell which parts?
}

impl Default for YOLOPredsFormat {
    fn default() -> Self {
        Self {
            clss: ClssType::Clss,
            bbox: None,
            kpts: None,
            coefs: None,
            obb: None,
            anchors: None,
            is_bbox_normalized: false,
            apply_nms: true,
            apply_softmax: false,
            // is_concatenated: true,
        }
    }
}

impl YOLOPredsFormat {
    pub fn apply_nms(mut self, x: bool) -> Self {
        self.apply_nms = x;
        self
    }

    pub fn apply_softmax(mut self, x: bool) -> Self {
        self.apply_softmax = x;
        self
    }

    pub fn n_clss() -> Self {
        // Classification: NClss
        Self {
            clss: ClssType::Clss,
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_confclss() -> Self {
        // YOLOv5 | YOLOv6 | YOLOv7 | YOLOX : NACxcywhConfClss
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::ConfClss,
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_confclss_coefs() -> Self {
        // YOLOv5 Segment : NACxcywhConfClssCoefs
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::ConfClss,
            coefs: Some(true),
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_a() -> Self {
        // YOLOv8 | YOLOv9 : NCxcywhClssA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn n_a_xyxy_confcls() -> Self {
        // YOLOv10 : NAXyxyConfCls
        Self {
            bbox: Some(BoxType::Xyxy),
            clss: ClssType::ConfCls,
            anchors: Some(AnchorsPosition::Before),
            ..Default::default()
        }
    }

    pub fn n_a_cxcywh_clss_n() -> Self {
        // RTDETR
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            anchors: Some(AnchorsPosition::Before),
            is_bbox_normalized: true,
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_xycs_a() -> Self {
        // YOLOv8 Pose : NCxcywhClssXycsA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            kpts: Some(KptsType::Xycs),
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_coefs_a() -> Self {
        // YOLOv8 Segment : NCxcywhClssCoefsA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            coefs: Some(true),
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn n_cxcywh_clss_r_a() -> Self {
        // YOLOv8 Obb : NCxcywhClssRA
        Self {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            obb: Some(true),
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    }

    pub fn task(&self) -> Task {
        match self.obb {
            Some(_) => Task::OrientedObjectDetection,
            None => match self.coefs {
                Some(_) => Task::InstanceSegmentation,
                None => match self.kpts {
                    Some(_) => Task::KeypointsDetection,
                    None => match self.bbox {
                        Some(_) => Task::ObjectDetection,
                        None => Task::ImageClassification,
                    },
                },
            },
        }
    }

    pub fn box_type(&self) -> Option<&BoxType> {
        self.bbox.as_ref()
    }

    pub fn is_anchors_first(&self) -> bool {
        matches!(self.anchors, Some(AnchorsPosition::Before))
    }

    pub fn is_cls_type(&self) -> bool {
        matches!(self.clss, ClssType::ClsConf | ClssType::ConfCls)
    }

    pub fn is_clss_type(&self) -> bool {
        matches!(
            self.clss,
            ClssType::ClssConf | ClssType::ConfClss | ClssType::Clss
        )
    }

    pub fn is_conf_at_end(&self) -> bool {
        matches!(self.clss, ClssType::ClssConf | ClssType::ClsConf)
    }

    pub fn is_conf_independent(&self) -> bool {
        !matches!(self.clss, ClssType::Clss)
    }

    pub fn kpt_step(&self) -> Option<usize> {
        match &self.kpts {
            Some(x) => match x {
                KptsType::Xycs => Some(3),
                KptsType::Xys => Some(2),
            },
            None => None,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn parse_preds(
        &self,
        x: TensorView,
        nc: usize,
    ) -> anyhow::Result<(
        Option<crate::Tensor>,
        Option<crate::Tensor>,
        crate::Tensor,
        Option<crate::Tensor>,
        Option<crate::Tensor>,
        Option<crate::Tensor>,
        Option<crate::Tensor>,
    )> {
        match self.task() {
            Task::ImageClassification => Ok((None, None, x.to_owned()?, None, None, None, None)),
            _ => {
                let x = if self.is_anchors_first() {
                    x.to_owned()?
                } else {
                    x.reversed_axes()?
                };

                // each tasks slices
                let x_view = x.view();
                let (slice_bboxes, xs_tensor) = x_view.split_at(1, 4)?;
                let xs_view = xs_tensor.view();

                let (slice_id, slice_clss, slice_confs, xs_tensor) = match self.clss {
                    ClssType::ConfClss => {
                        let (confs, xs) = xs_view.split_at(1, 1)?;
                        let (clss, xs) = xs.split_at(1, nc)?;
                        (None, Some(clss), Some(confs), xs)
                    }
                    ClssType::ClssConf => {
                        let (clss, xs) = xs_view.split_at(1, nc)?;
                        let (confs, xs) = xs.split_at(1, 1)?;
                        (None, Some(clss), Some(confs), xs)
                    }
                    ClssType::ConfCls => {
                        let (clss, xs) = xs_view.split_at(1, 1)?;
                        let (ids, xs) = xs.split_at(1, 1)?;
                        (Some(ids), Some(clss), None, xs)
                    }
                    ClssType::ClsConf => {
                        let (ids, xs) = xs_view.split_at(1, 1)?;
                        let (clss, xs) = xs.split_at(1, 1)?;
                        (Some(ids), Some(clss), None, xs)
                    }
                    ClssType::Clss => {
                        let (clss, xs) = xs_view.split_at(1, nc)?;
                        (None, Some(clss), None, xs)
                    }
                };
                let xs = xs_tensor.view();
                let (slice_kpts, slice_coefs, slice_radians) = match self.task() {
                    Task::Pose | Task::KeypointsDetection => {
                        // For pose detection, remaining xs contains keypoints
                        (Some(xs.to_owned()?), None, None)
                    }
                    Task::InstanceSegmentation => {
                        // For instance segmentation, remaining xs contains mask coefficients
                        (None, Some(xs.to_owned()?), None)
                    }
                    Task::Obb | Task::OrientedObjectDetection => {
                        // For OBB, remaining xs contains rotation angles
                        (None, None, Some(xs.to_owned()?))
                    }
                    _ => (None, None, None),
                };
                let xs_owned = xs.to_owned()?;

                Ok((
                    Some(slice_bboxes),
                    slice_id,
                    slice_clss.unwrap_or(xs_owned),
                    slice_confs,
                    slice_kpts,
                    slice_coefs,
                    slice_radians,
                ))
            }
        }
    }
}
