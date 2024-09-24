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
        "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4", // remote video
                                                                                             // "rtsp://admin:xyz@192.168.2.217:554/h265/ch1/",  // rtsp h264 stream
                                                                                             // "./assets/bus.jpg", // local image
                                                                                             // "/home/qweasd/Desktop/SourceVideos/3.mp4",
                                                                                             // "../hall.mp4",
                                                                                             // "../000020489.jpg",
    )?
    .with_batch(1)
    .build()?;

    let mut viewer = Viewer::new();

    // run
    for (xs, _) in dl {
        // std::thread::sleep(std::time::Duration::from_millis(100));
        let ys = model.forward(&xs, false)?;
        // annotator.annotate(&xs, &ys);

        // TODO: option for saving
        let images_plotted = annotator.plot(&xs, &ys)?;

        // RgbaImage -> DynamicImage
        let frames: Vec<image::DynamicImage> = images_plotted
            .iter()
            .map(|x| image::DynamicImage::from(x.to_owned()))
            .collect();

        // imshow
        viewer.imshow(&frames)?;
        if !viewer.is_open() || viewer.is_key_pressed(Key::Escape) {
            break;
        }

        // write video
        viewer.write_batch(&frames, 30)?; // fps
    }

    // images -> video
    // DataLoader::is2v("runs/YOLO-DataLoader", &["runs", "is2v"], 24)?;

    viewer.finish_write()?;

    Ok(())
}
