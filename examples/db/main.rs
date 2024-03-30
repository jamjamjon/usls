use usls::{models::DB, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/ppocr-v4-db-dyn.onnx")
        .with_i00((1, 1, 4).into())
        .with_i02((608, 640, 960).into())
        .with_i03((608, 640, 960).into())
        .with_confs(&[0.7])
        .with_saveout("DB-Text-Detection")
        .with_dry_run(5)
        // .with_trt(0)
        // .with_fp16(true)
        .with_profile(true);
    let mut model = DB::new(&options)?;

    // load image
    let x = DataLoader::try_read("./assets/math.jpg")?;

    // run
    let _y = model.run(&[x])?;

    Ok(())
}
