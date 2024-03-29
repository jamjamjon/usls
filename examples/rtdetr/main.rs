use usls::{models::RTDETR, DataLoader, Options, COCO_NAMES_80};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/rtdetr-l-f16.onnx")
        .with_confs(&[0.4, 0.15]) // person: 0.4, others: 0.15
        .with_names(&COCO_NAMES_80)
        .with_saveout("RT-DETR");
    let mut model = RTDETR::new(&options)?;

    // build dataloader
    let mut dl = DataLoader::default().load("./assets/bus.jpg")?;

    // run
    model.run(&dl.next().unwrap().0)?;

    Ok(())
}
