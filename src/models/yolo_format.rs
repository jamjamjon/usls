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
    V8,
    V9,
    V10,
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

    // Detections
    // Batch - Anchors - Bbox - Clss
    NACxcywhClss,
    NACxcyxyClss,
    NAXyxyClss,
    NAXywhClss,

    // Batch - Bbox - Clss - Anchors
    NCxcywhClssA,
    NCxcyxyClssA,
    NXyxyClssA,
    NXywhClssA,

    // Batch - Anchors - Bbox - Conf - Clss
    NACxcywhConfClss,
    NACxcyxyConfClss,
    NAXyxyConfClss,
    NAXywhConfClss,

    // Batch - Bbox - Conf - Clss - Anchors
    NCxcywhConfClssA,
    NCxcyxyConfClssA,
    NXyxyConfClssA,
    NXywhConfClssA,

    // Batch - Anchors - Bbox - Conf - Cls
    NACxcywhConfCls,
    NACxcyxyConfCls,
    NAXyxyConfCls,
    NAXywhConfCls,

    // Batch - Bbox - Conf - Cls - Anchors
    NCxcywhConfClsA,
    NCxcyxyConfClsA,
    NXyxyConfClsA,
    NXywhConfClsA,

    // anchor first, one top class, Confidence Independent
    // Batch - Anchors - Bbox - Cls - Conf
    NACxcywhClsConf,
    NACxcyxyClsConf,
    NAXyxyClsConf,
    NAXywhClsConf,

    // anchor later, one top class, Confidence Independent
    // Batch - Bbox - Cls - Conf - Anchors
    NCxcywhClsConfA,
    NCxcyxyClsConfA,
    NXyxyClsConfA,
    NXywhClsConfA,

    // Batch - Anchors - Bbox - Clss - Conf
    NACxcywhClssConf,
    NACxcyxyClssConf,
    NAXyxyClssConf,
    NAXywhClssConf,

    // Batch - Bbox - Clss - Conf - Anchors
    NCxcywhClssConfA,
    NCxcyxyClssConfA,
    NXyxyClssConfA,
    NXywhClssConfA,

    // ===> TODO: Keypoints: Xycs/Xys must be at the end
    // xys => xy, xy, ..., No keypoint confidence
    // xycs => xyc, xyc, ..., Has keypoint confidence
    NACxcywhClssXys,
    NACxcywhClssXycs,
    NACxcyxyClssXycs,
    NAXyxyClssXycs,
    NCxcywhClssXycsA,

    // ===> TODO: OBB
    NCxcywhClssRA, // R => radians
    NACxcywhClssR, // R => radians

    // ===> TODO: instance segment
    NCxcywhClssCoefsA,
    NACxcywhClssCoefs,
    NACxcywhConfClssCoefs,
    NCxcywhConfClssCoefsA,
}

impl fmt::Display for YOLOFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl YOLOFormat {
    pub fn box_type(&self) -> BoxType {
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
        !self.to_string().ends_with('A')
        // match self {
        //     YOLOFormat::NACxcywhClss
        //     | YOLOFormat::NACxcyxyClss
        //     | YOLOFormat::NAXyxyClss
        //     _ => false,
        // }
    }

    pub fn is_conf_independent(&self) -> bool {
        self.to_string().contains("Conf")
        // matches!(
        //     self,
        //     YOLOFormat::NAXywhConfCls
        //         | YOLOFormat::NACxcywhClsConf
        //         | YOLOFormat::NXyxyClsConfA
        //         | YOLOFormat::NXywhClsConfA
        //         | YOLOFormat::NACxcywhConfClssCoefs
        //         | YOLOFormat::NCxcywhConfClssCoefsA
        // )
    }

    pub fn is_conf_last(&self) -> bool {
        let s = self.to_string();
        let pos_conf = s.find("Conf").unwrap();
        let pos_clss = s.find("Cls").unwrap();
        pos_conf > pos_clss
        // matches!(
        //     self,
        //     YOLOFormat::NACxcywhClsConf
        //         | YOLOFormat::NACxcyxyClsConf
        //         | YOLOFormat::NAXyxyClsConf
        //         | YOLOFormat::NAXywhClsConf
        //         | YOLOFormat::NCxcywhClsConfA
        //         | YOLOFormat::NCxcyxyClsConfA
        //         | YOLOFormat::NXyxyClsConfA
        //         | YOLOFormat::NXywhClsConfA
        // )
    }

    pub fn is_cls(&self) -> bool {
        matches!(
            self,
            YOLOFormat::NAXywhConfCls
                | YOLOFormat::NACxcywhClsConf
                | YOLOFormat::NACxcyxyClsConf
                | YOLOFormat::NAXyxyClsConf
                | YOLOFormat::NAXywhClsConf
                | YOLOFormat::NACxcywhConfCls
                | YOLOFormat::NACxcyxyConfCls
                | YOLOFormat::NAXyxyConfCls
                | YOLOFormat::NCxcywhConfClsA
                | YOLOFormat::NCxcyxyConfClsA
                | YOLOFormat::NXyxyConfClsA
                | YOLOFormat::NXywhConfClsA
                | YOLOFormat::NCxcywhClsConfA
                | YOLOFormat::NCxcyxyClsConfA
                | YOLOFormat::NXyxyClsConfA
                | YOLOFormat::NXywhClsConfA
        )
    }

    pub fn is_clss(&self) -> bool {
        !self.is_cls()
    }

    pub fn is_cxcywh(&self) -> bool {
        let s = format!("{:?}", self);
        s.contains("Cxcywh")
        // matches!(
        //     self,
        //     YOLOFormat::NACxcywhClsConf
        //     | YOLOFormat::NACxcywhConfCls
        //     | YOLOFormat::NACxcywhClss
        //     | YOLOFormat::NACxcywhConfClss
        //     | YOLOFormat::NACxcywhClssXycs  // kpt
        //     | YOLOFormat::NCxcywhClssA
        //     | YOLOFormat::NCxcywhConfClssA
        //     | YOLOFormat::NCxcywhConfClsA
        //     | YOLOFormat::NCxcywhClsConfA
        //     | YOLOFormat::NCxcywhClssXycsA // kpt
        //     | YOLOFormat::NCxcywhClssCoefsA
        //     | YOLOFormat::NACxcywhClssCoefs
        //     | YOLOFormat::NACxcywhConfClssCoefs
        //     | YOLOFormat::NCxcywhConfClssCoefsA
        //     | YOLOFormat::NACxcywhClssR
        //     | YOLOFormat::NCxcywhClssRA
        // )
    }

    pub fn is_xywh(&self) -> bool {
        matches!(
            self,
            YOLOFormat::NAXywhConfCls
                | YOLOFormat::NAXywhClsConf
                | YOLOFormat::NAXywhClss
                | YOLOFormat::NAXywhConfClss
                | YOLOFormat::NXywhClssA
                | YOLOFormat::NXywhConfClssA
                | YOLOFormat::NXywhConfClsA
                | YOLOFormat::NXywhClsConfA
        )
    }

    pub fn is_cxcyxy(&self) -> bool {
        matches!(
            self,
            YOLOFormat::NACxcyxyClsConf
            | YOLOFormat::NACxcyxyConfCls
            | YOLOFormat::NACxcyxyConfClss
            // | YOLOFormat::NACxcyxyClssConf // TODO
            | YOLOFormat::NCxcyxyClssA
            | YOLOFormat::NCxcyxyConfClssA
            | YOLOFormat::NCxcyxyConfClsA
            | YOLOFormat::NCxcyxyClsConfA
        )
    }

    pub fn is_xyxy(&self) -> bool {
        matches!(
            self,
            YOLOFormat::NAXyxyClsConf
                | YOLOFormat::NAXyxyConfCls
                | YOLOFormat::NAXyxyClss
                | YOLOFormat::NAXyxyConfClss
                | YOLOFormat::NXyxyClssA
                | YOLOFormat::NXyxyConfClssA
                | YOLOFormat::NXyxyConfClsA
                | YOLOFormat::NXyxyClsConfA
        )
    }

    pub fn task(&self) -> YOLOTask {
        // TODO:
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
    pub fn is_clssification_task(&self) -> bool {
        matches!(self, YOLOFormat::NClss)
    }

    pub fn is_obb_task(&self) -> bool {
        matches!(self, YOLOFormat::NACxcywhClssR | YOLOFormat::NCxcywhClssRA)
    }

    pub fn is_kpt_task(&self) -> bool {
        matches!(
            self,
            YOLOFormat::NACxcywhClssXycs | YOLOFormat::NCxcywhClssXycsA
        )
    }

    pub fn is_seg_task(&self) -> bool {
        matches!(
            self,
            YOLOFormat::NCxcywhClssCoefsA
                | YOLOFormat::NACxcywhClssCoefs
                | YOLOFormat::NACxcywhConfClssCoefs
                | YOLOFormat::NCxcywhConfClssCoefsA
        )
    }

    pub fn kpt_step(&self) -> Option<usize> {
        match self {
            YOLOFormat::NACxcywhClssXys => Some(2),
            YOLOFormat::NACxcywhClssXycs
            | YOLOFormat::NACxcyxyClssXycs
            | YOLOFormat::NAXyxyClssXycs
            | YOLOFormat::NCxcywhClssXycsA => Some(3),
            _ => None,
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
        let (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians) = if self.is_cls() {
            // box-[cls | conf -[kpts | coefs]]
            if self.is_conf_last() {
                // box-cls-conf-[kpts | coefs]

                let (ids, xs) = xs.split_at(Axis(1), 1);
                let (clss, xs) = xs.split_at(Axis(1), 1);
                let slice_id = Some(ids);
                let slice_clss = clss.to_owned();

                let (slice_kpts, slice_coefs, slice_radians) = if self.is_kpt_task() {
                    (Some(xs), None, None)
                } else if self.is_seg_task() {
                    (None, Some(xs), None)
                } else if self.is_obb_task() {
                    (None, None, Some(xs))
                } else {
                    (None, None, None)
                };
                (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
            } else {
                // box-conf-cls-[kpts | coefs]

                let (clss, xs) = xs.split_at(Axis(1), 1);
                let (ids, xs) = xs.split_at(Axis(1), 1);
                let slice_id = Some(ids);
                let slice_clss = clss.to_owned();

                let (slice_kpts, slice_coefs, slice_radians) = if self.is_kpt_task() {
                    (Some(xs), None, None)
                } else if self.is_seg_task() {
                    (None, Some(xs), None)
                } else if self.is_obb_task() {
                    (None, None, Some(xs))
                } else {
                    (None, None, None)
                };
                (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
            }
        } else {
            // box-[clss | conf -[kpts | coefs]]
            if self.is_conf_independent() {
                if self.is_conf_last() {
                    // box-clss-conf-[kpts | coefs]

                    let slice_id = None;
                    let (clss, xs) = xs.split_at(Axis(1), nc);
                    let (confs, xs) = xs.split_at(Axis(1), 1);
                    let confs = confs.broadcast((confs.shape()[0], nc)).unwrap();
                    let clss = &confs * &clss;
                    let slice_clss = clss;

                    let (slice_kpts, slice_coefs, slice_radians) = if self.is_kpt_task() {
                        (Some(xs), None, None)
                    } else if self.is_seg_task() {
                        (None, Some(xs), None)
                    } else if self.is_obb_task() {
                        (None, None, Some(xs))
                    } else {
                        (None, None, None)
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
                    let (slice_kpts, slice_coefs, slice_radians) = if self.is_kpt_task() {
                        (Some(xs), None, None)
                    } else if self.is_seg_task() {
                        (None, Some(xs), None)
                    } else if self.is_obb_task() {
                        (None, None, Some(xs))
                    } else {
                        (None, None, None)
                    };
                    (slice_id, slice_clss, slice_kpts, slice_coefs, slice_radians)
                }
            } else {
                // box-[clss -[kpts | coefs]]
                let slice_id = None;
                let (clss, xs) = xs.split_at(Axis(1), nc);
                let slice_clss = clss.to_owned();

                let (slice_kpts, slice_coefs, slice_radians) = if self.is_kpt_task() {
                    (Some(xs), None, None)
                } else if self.is_seg_task() {
                    (None, Some(xs), None)
                } else if self.is_obb_task() {
                    (None, None, Some(xs))
                } else {
                    (None, None, None)
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
