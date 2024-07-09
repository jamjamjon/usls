use clap::ValueEnum;
use ndarray::{Array, ArrayBase, ArrayView, Axis, Dim, IxDyn, IxDynImpl, ViewRepr};
use std::fmt;

#[derive(Debug, Clone)]
pub enum BoxType {
    Cxcywh,
    Cxcyxy,
    Xyxy,
    Xywh,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum YOLOTask {
    Classify,
    Detect,
    Pose,
    Segment,
    Obb,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum YOLOVersion {
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    // TODO:
    // YOLOX,
    // YOLOv3,
    // YOLOv4,
}

/// Enumeration of various YOLO output formats.
///
/// This enum captures different possible output formats for YOLO models.
/// Here, `N` represents the batch size, `A` represents the anchors, `Conf`
/// represents the confidence or objectness, `clss` represents multiple
/// class conditional probabilities, and `cls` represents a single top-probability class.
#[derive(Debug, Clone, PartialEq, ValueEnum)]
pub enum YOLOFormat {
    /// Classification
    NClss,

    /// Detections: Batch - Anchors - Bbox - Clss
    NACxcywhClss,
    NACxcyxyClss,
    NAXyxyClss,
    NAXywhClss,

    /// Detections: Batch - Bbox - Clss - Anchors
    NCxcywhClssA,
    NCxcyxyClssA,
    NXyxyClssA,
    NXywhClssA,

    /// Detections: Batch - Anchors - Bbox - Conf - Clss
    NACxcywhConfClss,
    NACxcyxyConfClss,
    NAXyxyConfClss,
    NAXywhConfClss,

    /// Detections: Batch - Bbox - Conf - Clss - Anchors
    NCxcywhConfClssA,
    NCxcyxyConfClssA,
    NXyxyConfClssA,
    NXywhConfClssA,

    /// Detections: Batch - Anchors - Bbox - Conf - Cls
    NACxcywhConfCls,
    NACxcyxyConfCls,
    NAXyxyConfCls,
    NAXywhConfCls,

    /// Detections: Batch - Bbox - Conf - Cls - Anchors
    NCxcywhConfClsA,
    NCxcyxyConfClsA,
    NXyxyConfClsA,
    NXywhConfClsA,

    /// Detections: Batch - Anchors - Bbox - Cls - Conf
    NACxcywhClsConf,
    NACxcyxyClsConf,
    NAXyxyClsConf,
    NAXywhClsConf,

    /// Detections: Batch - Bbox - Cls - Conf - Anchors
    NCxcywhClsConfA,
    NCxcyxyClsConfA,
    NXyxyClsConfA,
    NXywhClsConfA,

    /// Detections:  Batch - Anchors - Bbox - Clss - Conf
    NACxcywhClssConf,
    NACxcyxyClssConf,
    NAXyxyClssConf,
    NAXywhClssConf,

    /// Detections: Batch - Bbox - Clss - Conf - Anchors
    NCxcywhClssConfA,
    NCxcyxyClssConfA,
    NXyxyClssConfA,
    NXywhClssConfA,

    // xys => xy, xy, ..., No keypoint confidence
    // xycs => xyc, xyc, ..., Has keypoint confidence
    /// Keypoints:  Batch - Anchors - Bbox - Clss - Xys
    NACxcywhClssXys,
    NACxcyxyClssXys,
    NAXyxyClssXys,
    NAXywhClssXys,

    /// Keypoints: Batch - Bbox - Clss - Anchors - Xys
    NCxcywhClssXysA,
    NCxcyxyClssXysA,
    NXyxyClssXysA,
    NXywhClssXysA,

    /// Keypoints: Batch - Anchors - Bbox - Conf - Clss - Xys
    NACxcywhConfClssXys,
    NACxcyxyConfClssXys,
    NAXyxyConfClssXys,
    NAXywhConfClssXys,

    /// Keypoints: Batch - Bbox - Conf - Clss - Anchors - Xys
    NCxcywhConfClssXysA,
    NCxcyxyConfClssXysA,
    NXyxyConfClssXysA,
    NXywhConfClssXysA,

    /// Keypoints: Batch - Anchors - Bbox - Conf - Cls - Xys
    NACxcywhConfClsXys,
    NACxcyxyConfClsXys,
    NAXyxyConfClsXys,
    NAXywhConfClsXys,

    /// Keypoints: Batch - Bbox - Conf - Cls - Anchors - Xys
    NCxcywhConfClsXysA,
    NCxcyxyConfClsXysA,
    NXyxyConfClsXysA,
    NXywhConfClsXysA,

    /// Keypoints: Batch - Anchors - Bbox - Cls - Conf - Xys
    NACxcywhClsConfXys,
    NACxcyxyClsConfXys,
    NAXyxyClsConfXys,
    NAXywhClsConfXys,

    /// Keypoints: Batch - Bbox - Cls - Conf - Anchors - Xys
    NCxcywhClsConfXysA,
    NCxcyxyClsConfXysA,
    NXyxyClsConfXysA,
    NXywhClsConfXysA,

    /// Keypoints: Batch - Anchors - Bbox - Clss - Conf - Xys
    NACxcywhClssConfXys,
    NACxcyxyClssConfXys,
    NAXyxyClssConfXys,
    NAXywhClssConfXys,

    /// Keypoints: Batch - Bbox - Clss - Conf - Anchors - Xys
    NCxcywhClssConfXysA,
    NCxcyxyClssConfXysA,
    NXyxyClssConfXysA,
    NXywhClssConfXysA,

    /// Keypoints: Batch - Anchors - Bbox - Clss - Xycs
    NACxcywhClssXycs,
    NACxcyxyClssXycs,
    NAXyxyClssXycs,
    NAXywhClssXycs,

    /// Keypoints: Batch - Bbox - Clss - Anchors - Xycs
    NCxcywhClssXycsA,
    NCxcyxyClssXycsA,
    NXyxyClssXycsA,
    NXywhClssXycsA,

    /// Keypoints: Batch - Anchors - Bbox - Conf - Clss - Xycs
    NACxcywhConfClssXycs,
    NACxcyxyConfClssXycs,
    NAXyxyConfClssXycs,
    NAXywhConfClssXycs,

    /// Keypoints: Batch - Bbox - Conf - Clss - Anchors - Xycs
    NCxcywhConfClssXycsA,
    NCxcyxyConfClssXycsA,
    NXyxyConfClssXycsA,
    NXywhConfClssXycsA,

    /// Keypoints: Batch - Anchors - Bbox - Conf - Cls - Xycs
    NACxcywhConfClsXycs,
    NACxcyxyConfClsXycs,
    NAXyxyConfClsXycs,
    NAXywhConfClsXycs,

    /// Keypoints: Batch - Bbox - Conf - Cls - Anchors - Xycs
    NCxcywhConfClsXycsA,
    NCxcyxyConfClsXycsA,
    NXyxyConfClsXycsA,
    NXywhConfClsXycsA,

    /// Keypoints: Batch - Anchors - Bbox - Cls - Conf - Xycs
    NACxcywhClsConfXycs,
    NACxcyxyClsConfXycs,
    NAXyxyClsConfXycs,
    NAXywhClsConfXycs,

    // anchor later, one top class, Confidence Independent - Xycs
    /// Keypoints: Batch - Bbox - Cls - Conf - Anchors - Xycs
    NCxcywhClsConfXycsA,
    NCxcyxyClsConfXycsA,
    NXyxyClsConfXycsA,
    NXywhClsConfXycsA,

    /// Keypoints: Batch - Anchors - Bbox - Clss - Conf - Xycs
    NACxcywhClssConfXycs,
    NACxcyxyClssConfXycs,
    NAXyxyClssConfXycs,
    NAXywhClssConfXycs,

    /// Keypoints: Batch - Bbox - Clss - Conf - Anchors - Xycs
    NCxcywhClssConfXycsA,
    NCxcyxyClssConfXycsA,
    NXyxyClssConfXycsA,
    NXywhClssConfXycsA,

    // R => radians
    /// OBB: Batch - Anchors - Bbox - Clss - R
    NACxcywhClssR,
    NACxcyxyClssR,
    NAXyxyClssR,
    NAXywhClssR,

    /// OBB: Batch - Bbox - Clss - Anchors - R
    NCxcywhClssRA,
    NCxcyxyClssRA,
    NXyxyClssRA,
    NXywhClssRA,

    /// OBB: Batch - Anchors - Bbox - Conf - Clss - R
    NACxcywhConfClssR,
    NACxcyxyConfClssR,
    NAXyxyConfClssR,
    NAXywhConfClssR,

    /// OBB: Batch - Bbox - Conf - Clss - Anchors - R
    NCxcywhConfClssRA,
    NCxcyxyConfClssRA,
    NXyxyConfClssRA,
    NXywhConfClssRA,

    /// OBB: Batch - Anchors - Bbox - Conf - Cls - R
    NACxcywhConfClsR,
    NACxcyxyConfClsR,
    NAXyxyConfClsR,
    NAXywhConfClsR,

    /// OBB: Batch - Bbox - Conf - Cls - Anchors - R
    NCxcywhConfClsRA,
    NCxcyxyConfClsRA,
    NXyxyConfClsRA,
    NXywhConfClsRA,

    /// OBB: Batch - Anchors - Bbox - Cls - Conf - R
    NACxcywhClsConfR,
    NACxcyxyClsConfR,
    NAXyxyClsConfR,
    NAXywhClsConfR,

    /// OBB: Batch - Bbox - Cls - Conf - Anchors - R
    NCxcywhClsConfRA,
    NCxcyxyClsConfRA,
    NXyxyClsConfRA,
    NXywhClsConfRA,

    /// OBB: Batch - Anchors - Bbox - Clss - Conf - R
    NACxcywhClssConfR,
    NACxcyxyClssConfR,
    NAXyxyClssConfR,
    NAXywhClssConfR,

    /// OBB: Batch - Bbox - Clss - Conf - Anchors - R
    NCxcywhClssConfRA,
    NCxcyxyClssConfRA,
    NXyxyClssConfRA,
    NXywhClssConfRA,

    /// Instance Segment: Batch - Anchors - Bbox - Clss - Coefs
    NACxcywhClssCoefs,
    NACxcyxyClssCoefs,
    NAXyxyClssCoefs,
    NAXywhClssCoefs,

    /// Instance Segment: Batch - Bbox - Clss - Anchors - Coefs
    NCxcywhClssCoefsA,
    NCxcyxyClssCoefsA,
    NXyxyClssCoefsA,
    NXywhClssCoefsA,

    /// Instance Segment: Batch - Anchors - Bbox - Conf - Clss - Coefs
    NACxcywhConfClssCoefs,
    NACxcyxyConfClssCoefs,
    NAXyxyConfClssCoefs,
    NAXywhConfClssCoefs,

    /// Instance Segment: Batch - Bbox - Conf - Clss - Anchors - Coefs
    NCxcywhConfClssCoefsA,
    NCxcyxyConfClssCoefsA,
    NXyxyConfClssCoefsA,
    NXywhConfClssCoefsA,

    /// Instance Segment: Batch - Anchors - Bbox - Conf - Cls - Coefs
    NACxcywhConfClsCoefs,
    NACxcyxyConfClsCoefs,
    NAXyxyConfClsCoefs,
    NAXywhConfClsCoefs,

    /// Instance Segment: Batch - Bbox - Conf - Cls - Anchors - Coefs
    NCxcywhConfClsCoefsA,
    NCxcyxyConfClsCoefsA,
    NXyxyConfClsCoefsA,
    NXywhConfClsCoefsA,

    /// Instance Segment: Batch - Anchors - Bbox - Cls - Conf - Coefs
    NACxcywhClsConfCoefs,
    NACxcyxyClsConfCoefs,
    NAXyxyClsConfCoefs,
    NAXywhClsConfCoefs,

    /// Instance Segment: Batch - Bbox - Cls - Conf - Anchors - Coefs
    NCxcywhClsConfCoefsA,
    NCxcyxyClsConfCoefsA,
    NXyxyClsConfCoefsA,
    NXywhClsConfCoefsA,

    /// Instance Segment: Batch - Anchors - Bbox - Clss - Conf - Coefs
    NACxcywhClssConfCoefs,
    NACxcyxyClssConfCoefs,
    NAXyxyClssConfCoefs,
    NAXywhClssConfCoefs,

    /// Instance Segment: Batch - Bbox - Clss - Conf - Anchors - Coefs
    NCxcywhClssConfCoefsA,
    NCxcyxyClssConfCoefsA,
    NXyxyClssConfCoefsA,
    NXywhClssConfCoefsA,
}

impl fmt::Display for YOLOFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl YOLOFormat {
    pub fn box_type(&self) -> BoxType {
        // TODO: matches!
        let s = self.to_string();
        if s.contains("Cxcywh") {
            BoxType::Cxcywh
        } else if s.contains("Cxcyxy") {
            BoxType::Cxcyxy
        } else if s.contains("Xyxy") {
            BoxType::Xyxy
        } else {
            BoxType::Xywh
        }
    }

    pub fn is_anchors_first(&self) -> bool {
        // TODO: matches!
        !self.to_string().ends_with('A')
    }

    pub fn is_conf_independent(&self) -> bool {
        // TODO: matches!
        self.to_string().contains("Conf")
    }

    pub fn is_conf_at_end(&self) -> bool {
        // TODO: matches!
        let s = self.to_string();
        let pos_conf = s.find("Conf").unwrap();
        let pos_clss = s.find("Cls").unwrap();
        pos_conf > pos_clss
    }

    pub fn is_cls_type(&self) -> bool {
        // TODO: matches!
        !self.is_clss_type()
    }

    pub fn is_clss_type(&self) -> bool {
        // TODO: matches!
        self.to_string().contains("Clss")
    }

    pub fn task(&self) -> YOLOTask {
        // TODO: matches!
        match self {
            YOLOFormat::NACxcywhClssXycs | YOLOFormat::NCxcywhClssXycsA => YOLOTask::Pose,
            YOLOFormat::NCxcywhClssCoefsA
            | YOLOFormat::NACxcywhClssCoefs
            | YOLOFormat::NACxcywhConfClssCoefs
            | YOLOFormat::NCxcywhConfClssCoefsA => YOLOTask::Segment,
            YOLOFormat::NClss => YOLOTask::Classify,
            YOLOFormat::NACxcywhClssR | YOLOFormat::NCxcywhClssRA => YOLOTask::Obb,
            _ => YOLOTask::Detect,
        }
    }
    // pub fn is_clssification_task(&self) -> bool {
    //     matches!(self, YOLOFormat::NClss)
    // }

    // pub fn is_obb_task(&self) -> bool {
    //     matches!(self, YOLOFormat::NACxcywhClssR | YOLOFormat::NCxcywhClssRA)
    // }

    // pub fn is_kpt_task(&self) -> bool {
    //     matches!(
    //         self,
    //         YOLOFormat::NACxcywhClssXycs | YOLOFormat::NCxcywhClssXycsA
    //     )
    // }

    // pub fn is_seg_task(&self) -> bool {
    //     matches!(
    //         self,
    //         YOLOFormat::NCxcywhClssCoefsA
    //             | YOLOFormat::NACxcywhClssCoefs
    //             | YOLOFormat::NACxcywhConfClssCoefs
    //             | YOLOFormat::NCxcywhConfClssCoefsA
    //     )
    // }

    pub fn kpt_step(&self) -> Option<usize> {
        // TODO: matches!
        let s = self.to_string();
        if s.contains("Xys") {
            Some(2)
        } else if s.contains("Xycs") {
            Some(3)
        } else {
            None
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn parse_preds<'a>(
        &'a self,
        preds: ArrayBase<ViewRepr<&'a f32>, Dim<IxDynImpl>>,
        nc: usize,
    ) -> (
        ArrayView<f32, IxDyn>,
        Option<ArrayView<f32, IxDyn>>,
        Array<f32, IxDyn>,
        Option<ArrayView<f32, IxDyn>>,
        Option<ArrayView<f32, IxDyn>>,
        Option<ArrayView<f32, IxDyn>>,
    ) {
        let preds = if self.is_anchors_first() {
            preds
        } else {
            preds.reversed_axes()
        };

        // get each tasks slices
        let (slice_bboxes, xs) = preds.split_at(Axis(1), 4);
        let (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians) = if self.is_cls_type() {
            // box-[cls | conf -[kpts | coefs]]
            if self.is_conf_at_end() {
                // box-cls-conf-[kpts | coefs]

                let (ids, xs) = xs.split_at(Axis(1), 1);
                let (clss, xs) = xs.split_at(Axis(1), 1);
                let slice_id = Some(ids);
                let slice_clss = clss.to_owned();

                let (slice_kpts, slice_coefs, slice_radians) = match self.task() {
                    YOLOTask::Pose => (Some(xs), None, None),
                    YOLOTask::Segment => (None, Some(xs), None),
                    YOLOTask::Obb => (None, None, Some(xs)),
                    _ => (None, None, None),
                };

                (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
            } else {
                // box-conf-cls-[kpts | coefs]

                let (clss, xs) = xs.split_at(Axis(1), 1);
                let (ids, xs) = xs.split_at(Axis(1), 1);
                let slice_id = Some(ids);
                let slice_clss = clss.to_owned();

                let (slice_kpts, slice_coefs, slice_radians) = match self.task() {
                    YOLOTask::Pose => (Some(xs), None, None),
                    YOLOTask::Segment => (None, Some(xs), None),
                    YOLOTask::Obb => (None, None, Some(xs)),
                    _ => (None, None, None),
                };
                (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
            }
        } else {
            // box-[clss | conf -[kpts | coefs]]
            if self.is_conf_independent() {
                if self.is_conf_at_end() {
                    // box-clss-conf-[kpts | coefs]

                    let slice_id = None;
                    let (clss, xs) = xs.split_at(Axis(1), nc);
                    let (confs, xs) = xs.split_at(Axis(1), 1);
                    let confs = confs.broadcast((confs.shape()[0], nc)).unwrap();
                    let clss = &confs * &clss;
                    let slice_clss = clss;

                    let (slice_kpts, slice_coefs, slice_radians) = match self.task() {
                        YOLOTask::Pose => (Some(xs), None, None),
                        YOLOTask::Segment => (None, Some(xs), None),
                        YOLOTask::Obb => (None, None, Some(xs)),
                        _ => (None, None, None),
                    };
                    (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
                } else {
                    // box-conf-clss-[kpts | coefs]
                    let slice_id = None;
                    let (confs, xs) = xs.split_at(Axis(1), 1);
                    let (clss, xs) = xs.split_at(Axis(1), nc);
                    let confs = confs.broadcast((confs.shape()[0], nc)).unwrap();
                    let clss = &confs * &clss;
                    let slice_clss = clss;

                    let (slice_kpts, slice_coefs, slice_radians) = match self.task() {
                        YOLOTask::Pose => (Some(xs), None, None),
                        YOLOTask::Segment => (None, Some(xs), None),
                        YOLOTask::Obb => (None, None, Some(xs)),
                        _ => (None, None, None),
                    };
                    (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
                }
            } else {
                // box-[clss -[kpts | coefs]]
                let slice_id = None;
                let (clss, xs) = xs.split_at(Axis(1), nc);
                let slice_clss = clss.to_owned();

                let (slice_kpts, slice_coefs, slice_radians) = match self.task() {
                    YOLOTask::Pose => (Some(xs), None, None),
                    YOLOTask::Segment => (None, Some(xs), None),
                    YOLOTask::Obb => (None, None, Some(xs)),
                    _ => (None, None, None),
                };
                (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
            }
        };
        (
            slice_bboxes,
            slice_id,
            slice_clss,
            slice_kpts,
            slice_coefs,
            slice_radians,
        )
    }
}
