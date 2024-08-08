use usls::{models::GroundingDINO, Annotator, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Options::default()
        .with_i00((1, 1, 4).into())
        .with_i02((640, 800, 1200).into())
        .with_i03((640, 1200, 1200).into())
        .with_i10((1, 1, 4).into())
        .with_i11((256, 256, 512).into())
        .with_i20((1, 1, 4).into())
        .with_i21((256, 256, 512).into())
        .with_i30((1, 1, 4).into())
        .with_i31((256, 256, 512).into())
        .with_i40((1, 1, 4).into())
        .with_i41((256, 256, 512).into())
        .with_i50((1, 1, 4).into())
        .with_i51((256, 256, 512).into())
        .with_i52((256, 256, 512).into())
        .with_model("groundingdino-swin-t-ogc.onnx")? // TODO: export one more robust onnx model
        .with_confs(&[0.22])
        // .with_cpu()
        .with_profile(false);
    let mut model = GroundingDINO::new(opts)?;

    // Load images and set class names
    let x = [DataLoader::try_read("./assets/bus.jpg")?];
    let texts = [
        "person", "hand", "shoes", "bus", "dog", "cat", "sign", "tie", "monitor", "window",
        "glasses", "tree",
    ];

    // Run and annotate
    let y = model.run(&x, &texts)?;
    let annotator = Annotator::default()
        .with_bboxes_thickness(4)
        .with_saveout("GroundingDINO");
    annotator.annotate(&x, &y);

    Ok(())
}
