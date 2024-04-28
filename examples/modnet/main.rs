use usls::{models::MODNet, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/modnet-dyn.onnx")
        .with_i00((1, 1, 4).into())
        .with_i02((416, 512, 800).into())
        .with_i03((416, 512, 800).into())
        .with_trt(0)
        .with_fp16(true);
    let model = MODNet::new(&options)?;

    // load image
    let x = vec![DataLoader::try_read("./assets/portrait.jpg")?];

    // run
    let y = model.run(&x)?;
    println!("{:?}", y);

    // annotate
    let annotator = Annotator::default()
        .with_colormap_turbo(false)
        .with_saveout("MODNet");
    annotator.annotate(&x, &y);

    Ok(())
}
