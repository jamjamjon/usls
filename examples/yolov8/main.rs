use usls::{coco, models::YOLO, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("yolov8m-dyn-f16.onnx")?
        // .with_model("../models/yolov8m-pose-dyn-f16.onnx")
        // .with_model("../models/yolov8m-seg-dyn-f16.onnx")
        // .with_model("../models/yolov8s-cls.onnx")
        // .with_model("../models/yolov8s-obb.onnx")
        // .with_trt(0)
        // .with_fp16(true)
        .with_i00((1, 1, 4).into())
        // .with_i02((224, 1024, 1024).into())
        // .with_i03((224, 1024, 1024).into())
        .with_i02((224, 640, 800).into())
        .with_i03((224, 640, 800).into())
        .with_confs(&[0.4, 0.15]) // person: 0.4, others: 0.15
        .with_names2(&coco::KEYPOINTS_NAMES_17)
        .with_profile(true)
        .with_dry_run(10);
    let mut model = YOLO::new(&options)?;

    // build dataloader
    let dl = DataLoader::default()
        .with_batch(1)
        .load("./assets/bus.jpg")?;
    // .load("./assets/dota.png")?;

    // build annotate
    let annotator = Annotator::default()
        // .with_probs_topk(10)
        // // bboxes
        // .without_bboxes(false)
        .without_bboxes_conf(true)
        // .without_bboxes_name(true)
        // .without_bboxes_text_bg(false)
        // .with_bboxes_text_color([255, 255, 255, 255])
        // .with_bboxes_text_bg_alpha(255)
        // // keypoints
        // .without_keypoints(false)
        // .with_keypoints_palette(&COCO_KEYPOINT_COLORS_17)
        .with_skeletons(&coco::SKELETONS_16)
        // .with_keypoints_name(false)
        // .with_keypoints_conf(false)
        // .without_keypoints_text_bg(false)
        // .with_keypoints_text_color([255, 255, 255, 255])
        // .with_keypoints_text_bg_alpha(255)
        // .with_keypoints_radius(4)
        // // masks
        // .without_masks(false)
        // .with_masks_alpha(190)
        // .without_polygons(false)
        // // .with_polygon_color([0, 255, 255, 255])
        // .with_masks_conf(false)
        // .with_masks_name(true)
        // .with_masks_text_bg(true)
        // .with_masks_text_color([255, 255, 255, 255])
        // .with_masks_text_bg_alpha(10)
        // // mbrs
        // .without_mbrs(false)
        // .without_mbrs_conf(false)
        // .without_mbrs_name(false)
        // .without_mbrs_text_bg(false)
        // .with_mbrs_text_color([255, 255, 255, 255])
        // .with_mbrs_text_bg_alpha(70)
        .with_saveout("YOLOv8");

    // run & annotate
    for (xs, _paths) in dl {
        let ys = model.run(&xs)?;
        annotator.annotate(&xs, &ys);
    }

    Ok(())
}
