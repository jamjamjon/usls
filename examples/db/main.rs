use usls::{models::DB, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_ixx(0, 0, (1, 4, 8).into())
        .with_ixx(0, 2, (608, 960, 1280).into())
        .with_ixx(0, 3, (608, 960, 1280).into())
        // .with_trt(0)
        .with_confs(&[0.4])
        .with_min_width(5.0)
        .with_min_height(12.0)
        .with_model("db/ppocr-v4-db-dyn.onnx")?;

    let mut model = DB::new(options)?;

    // load image
    let x = [
        DataLoader::try_read("images/db.png")?,
        DataLoader::try_read("images/street.jpg")?,
    ];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .without_bboxes(true)
        .with_polygons_alpha(60)
        .with_contours_color([255, 105, 180, 255])
        .without_mbrs(true)
        .with_saveout("DB");
    annotator.annotate(&x, &y);

    Ok(())
}
