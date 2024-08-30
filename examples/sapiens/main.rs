use usls::{
    models::{Sapiens, SapiensTask},
    Annotator, DataLoader, Options,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // options
    let options = Options::default()
        .with_model("../sapiens-models/sapiens-seg-0.3b-f16.onnx")? // TensorRT is supported
        // .with_model("../sapiens-models/sapiens-seg-0.3b-u8.onnx")?
        // .with_model("../sapiens-models/sapiens-seg-0.3b-q4f16.onnx")?
        .with_sapiens_task(SapiensTask::Seg)
        .with_trt(0)
        .with_fp16(true)
        .with_dry_run(5)
        .with_profile(true)
        .with_names(&[
            "Background",
            "Apparel",
            "Face Neck",
            "Hair",
            "Left Foot",
            "Left Hand",
            "Left Lower Arm",
            "Left Lower Leg",
            "Left Shoe",
            "Left Sock",
            "Left Upper Arm",
            "Left Upper Leg",
            "Lower Clothing",
            "Right Foot",
            "Right Hand",
            "Right Lower Arm",
            "Right Lower Leg",
            "Right Shoe",
            "Right Sock",
            "Right Upper Arm",
            "Right Upper Leg",
            "Torso",
            "Upper Clothing",
            "Lower Lip",
            "Upper Lip",
            "Lower Teeth",
            "Upper Teeth",
            "Tongue",
        ])
        .with_i00((1, 1, 8).into());
    let mut model = Sapiens::new(options)?;

    // load
    let x = [DataLoader::try_read("./assets/pexels.jpg")?];

    // run
    let y = model.run(&x)?;

    // annotate
    let annotator = Annotator::default()
        .without_masks(true)
        // .with_colormap("Inferno")
        .with_saveout("Sapiens");
    annotator.annotate(&x, &y);

    Ok(())
}
