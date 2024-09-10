// #![allow(unused)]

use usls::{models::YOLO, Annotator, DataLoader, Options, Vision, YOLOTask, YOLOVersion};

fn main() -> anyhow::Result<()> {
    let options = Options::default()
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
    let annotator = Annotator::default()
        .with_bboxes_thickness(4)
        .with_saveout("YOLO-Video-Stream");

    // build dataloader
    // let image = DataLoader::try_read("images/car.jpg")?;
    let dl = DataLoader::new(
        // "https://github.com/jamjamjon/assets/releases/download/images/bus.jpg",
        // "rtsp://admin:zfsoft888@192.168.2.217:554/h265/ch1/",
        // "rtsp://admin:KCNULU@192.168.2.193:554/h264/ch1/",
        // "/home/qweasd/Desktop/coco/val2017/images/test",
        // "../hall.mp4",
        // "./assets/bus.jpg",
        // "image/cat.jpg",
        // "../set-negs",
        "/home/qweasd/Desktop/SourceVideos/3.mp4", // 400-800 us
                                                   // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4", // ~2ms
    )?
    .with_batch(3)
    .build()?;

    for (xs, _paths) in dl {
        println!("xs: {:?} | {:?}", xs.len(), _paths);
        let ys = model.forward(&xs, false)?;
        annotator.annotate(&xs, &ys);
    }

    Ok(())
}
