use usls::{models::DB, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_i00((1, 4, 8).into())
        .with_i02((608, 960, 1280).into())
        .with_i03((608, 960, 1280).into())
        .with_confs(&[0.4])
        .with_min_width(5.0)
        .with_min_height(12.0)
        // .with_trt(0)
        .with_model("../models/ppocr-v4-db-dyn.onnx");
    let mut model = DB::new(&options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/db.png")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .with_polygon_color([255u8, 0u8, 0u8])
        .without_name(true)
        .without_polygons(false)
        .without_bboxes(false)
        .with_saveout("DB-Text-Detection");
    annotator.annotate(&x, &y);

    Ok(())
}
