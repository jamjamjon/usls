use usls::{
    models::{SamPrompt, SAM},
    Annotator, DataLoader, Options,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // encoder
    let options_encoder = Options::default()
        // .with_cpu()
        .with_i00((1, 1, 1).into())
        .with_model("mobile-sam-vit-t-encoder.onnx")?;

    // decoder
    let options_decoder = Options::default()
        // .with_cpu()
        .with_i11((1, 1, 1).into())
        .with_i21((1, 1, 1).into())
        .with_find_contours(true) // find contours or not
        .with_model("mobile-sam-vit-t-decoder.onnx")?;

    // build model
    let mut model = SAM::new(options_encoder, options_decoder)?;

    // build dataloader
    let dl = DataLoader::default()
        .with_batch(model.batch() as _)
        .load("./assets/truck.jpg")?;

    // build annotator
    let annotator = Annotator::default()
        .with_bboxes_thickness(7)
        .without_bboxes_name(true)
        .without_bboxes_conf(true)
        .without_mbrs(true)
        .with_saveout("SAM");

    // run & annotate
    for (xs, _paths) in dl {
        // prompt
        let prompts = vec![
            SamPrompt::default()
                // .with_postive_point(774., 366.),   // postive point
                // .with_negative_point(774., 366.),   // negative point
                .with_bbox(215., 297., 643., 459.), // bbox
        ];
        let ys = model.run(&xs, &prompts)?;
        annotator.annotate(&xs, &ys);
    }

    Ok(())
}
