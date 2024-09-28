use usls::{
    models::{Sapiens, SapiensTask},
    Annotator, DataLoader, Options, BODY_PARTS_28,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build
    let options = Options::default()
        .with_model("sapiens/seg-0.3b-dyn.onnx")?
        .with_sapiens_task(SapiensTask::Seg)
        .with_names(&BODY_PARTS_28);
    let mut model = Sapiens::new(options)?;

    // load
    let x = [DataLoader::try_read("images/paul-george.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .without_masks(true)
        .with_polygons_name(false)
        .with_saveout("Sapiens");
    annotator.annotate(&x, &y);

    Ok(())
}
