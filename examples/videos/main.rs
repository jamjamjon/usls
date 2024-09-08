use usls::{
    models::YOLO, Annotator, DataLoader, Options, Vision, YOLOTask, YOLOVersion, COCO_SKELETONS_16,
};

fn main() -> anyhow::Result<()> {
    let options = Options::default()
        .with_cuda(0)
        .with_model("yolo/v8-m-pose-dyn.onnx")?
        .with_yolo_version(YOLOVersion::V8)
        .with_yolo_task(YOLOTask::Pose)
        .with_i00((1, 1, 4).into())
        .with_i02((0, 640, 640).into())
        .with_i03((0, 640, 640).into())
        .with_confs(&[0.2, 0.15]);
    let mut model = YOLO::new(options)?;

    // build annotator
    let annotator = Annotator::default()
        .with_skeletons(&COCO_SKELETONS_16)
        .with_bboxes_thickness(4)
        .with_saveout("YOLO-Video-Stream");

    // build dataloader
    let dl = DataLoader::new(
        // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        // "rtsp://stream.pchome.com.tw/pc/pcmedia/1080p.mp4",
        // "rtsp://185.107.232.253:554/live/stream",
        // "/home/qweasd/Desktop/SourceVideos/3.mp4",
        "./assets/bus.jpg",
        // "/home/qweasd/Desktop/coco/val2017/images/test",
        // "https://github.com/jamjamjon/assets/releases/download/images/bus.jpg",
    )?
    .with_batch(1);

    // run
    for (xs, _paths) in dl {
        let ys = model.forward(&xs, false)?;
        annotator.annotate(&xs, &ys);
    }

    Ok(())
}
