use usls::{models::YOLO, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/FastSAM-s-dyn-f16.onnx")
        .with_i00((1, 1, 4).into())
        .with_i02((416, 640, 800).into())
        .with_i03((416, 640, 800).into())
        .with_confs(&[0.4, 0.15]) // person: 0.4, others: 0.15
        .with_saveout("FastSAM")
        .with_profile(false);
    let mut model = YOLO::new(&options)?;

    // build dataloader
    let mut dl = DataLoader::default().load("./assets/bus.jpg")?;

    // run
    model.run(&dl.next().unwrap().0)?;

    Ok(())
}
