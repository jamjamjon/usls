use usls::{models::YOLO, Annotator, DataLoader, Options, Vision, YOLOTask, YOLOVersion};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .init();

    let options = Options::new()
        .with_cuda(0)
        .with_model("yolo/v8-m-dyn.onnx")?
        .with_yolo_version(YOLOVersion::V8)
        .with_yolo_task(YOLOTask::Detect)
        .with_i00((1, 1, 4).into())
        .with_i02((0, 640, 640).into())
        .with_i03((0, 640, 640).into())
        .with_confs(&[0.2]);
    let mut model = YOLO::new(options)?;

    // build dataloader
    let dl = DataLoader::new(
        // "rtsp://admin:zfsoft888@192.168.2.217:554/h265/ch1/",
        // "rtsp://admin:KCNULU@192.168.2.193:554/h264/ch1/",
        // "../hall.mp4",
        "./assets/bus.jpg",
        // "images/car.jpg",
        // "../set-negs",
        // "/home/qweasd/Desktop/coco/val2017/images/test",
        // "/home/qweasd/Desktop/SourceVideos/3.mp4",
        // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    )?
    .with_batch(1)
    .with_progress_bar(true)
    .build()?;

    // // build annotator
    let annotator = Annotator::new()
        .with_bboxes_thickness(4)
        .with_saveout("YOLO-DataLoader");

    // run
    for (xs, _) in dl {
        // std::thread::sleep(std::time::Duration::from_millis(10000));
        let ys = model.forward(&xs, false)?;
        annotator.annotate(&xs, &ys);
    }

    // images -> video
    // DataLoader::is2v("runs/YOLO-DataLoader", &["runs", "is2v"], 24)?;

    Ok(())
}
