use usls::{
    models::YOLO, Annotator, DataLoader, Key, Options, Viewer, Vision, YOLOTask, YOLOVersion,
};

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

    // build annotator
    let annotator = Annotator::new()
        .with_bboxes_thickness(4)
        .with_saveout("YOLO-DataLoader");

    // build dataloader
    let dl = DataLoader::new(
        // "images/bus.jpg",  // remote image
        // "../images", // image folder
        // "../demo.mp4",   // local video
        // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4", // remote video
        // "rtsp://admin:xyz@192.168.2.217:554/h265/ch1/",  // rtsp h264 stream
        // "./assets/bus.jpg", // local image
        "../SourceVideos/3.mp4",
    )?
    .with_batch(1)
    .build()?;

    let mut viewer = Viewer::new().with_delay(100).with_scale(1.).resizable(true);

    // iteration
    for (xs, _) in dl {
        // inference & annotate
        let ys = model.run(&xs)?;
        let images_plotted = annotator.plot(&xs, &ys, false)?;

        // show image
        viewer.imshow(&images_plotted)?;

        // write video
        viewer.write_batch(&images_plotted)?;

        // check out window and key event
        if !viewer.is_open() || viewer.is_key_pressed(Key::Escape) {
            break;
        }
    }

    // finish video write
    viewer.finish_write()?;

    // images -> video
    // DataLoader::is2v("runs/YOLO-DataLoader", &["runs", "is2v"], 24)?;

    Ok(())
}
