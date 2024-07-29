use usls::{
    models::{SamKind, SamPrompt, SAM},
    Annotator, DataLoader, Options,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device_id = 0;
    let kind = SamKind::Sam;
    // let kind = SamKind::SamHq;
    // let kind = SamKind::EdgeSam;

    let (options_encoder, options_decoder) = match kind {
        SamKind::Sam | SamKind::MobileSam => {
            let options_encoder = Options::default()
                // .with_model("sam-vit-b-01ec64-encoder-qnt.onnx")?;
                // .with_model("sam-vit-b-01ec64-encoder.onnx")?;
                .with_model("mobile-sam-vit-t-encoder.onnx")?;

            let options_decoder = Options::default()
                .with_i00((1, 1, 1).into())
                .with_i11((1, 1, 1).into())
                .with_i21((1, 1, 1).into())
                .with_sam_kind(SamKind::Sam)
                // .with_model("sam-vit-b-01ec64-decoder-singlemask.onnx")?;
                // .with_model("sam-vit-b-01ec64-decoder.onnx")?;
                .with_model("mobile-sam-vit-t-decoder.onnx")?;

            (options_encoder, options_decoder)
        }
        SamKind::SamHq => {
            let options_encoder = Options::default()
                // .with_model("sam-hq-vit-b-encoder.onnx")?;
                .with_model("sam-hq-vit-t-encoder.onnx")?;

            let options_decoder = Options::default()
                .with_i00((1, 1, 1).into())
                .with_i21((1, 1, 1).into())
                .with_i31((1, 1, 1).into())
                .with_sam_kind(SamKind::SamHq)
                // .with_model("sam-hq-vit-b-decoder.onnx")?;
                .with_model("sam-hq-vit-t-decoder.onnx")?;
            (options_encoder, options_decoder)
        }
        SamKind::EdgeSam => {
            let options_encoder =
                Options::default().with_model("/home/qweasd/Downloads/edge-sam-3x-encoder.onnx")?;

            let options_decoder = Options::default()
                .with_i00((1, 1, 1).into())
                .with_i11((1, 1, 1).into())
                .with_i21((1, 1, 1).into())
                .with_sam_kind(SamKind::EdgeSam)
                .with_model("/home/qweasd/Downloads/edge-sam-3x-decoder.onnx")?;
            (options_encoder, options_decoder)
        }
    };
    let options_encoder = options_encoder
        .with_cuda(device_id)
        .with_i00((1, 1, 1).into())
        .with_i02((800, 1024, 1024).into())
        .with_i03((800, 1024, 1024).into());
    let options_decoder = options_decoder
        .with_cuda(device_id)
        .with_find_contours(true); // find contours or not

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
            SamPrompt::default().with_postive_point(500., 375.), // postive point
                                                                 // .with_postive_point(1125., 625.), // postive point
                                                                 // .with_postive_point(774., 366.), // postive point
                                                                 // .with_negative_point(774., 366.),   // negative point
                                                                 // .with_bbox(300., 175., 525., 500.), // bbox
                                                                 // .with_bbox(215., 297., 643., 459.), // bbox

                                                                 // .with_bbox(26., 20., 873., 990.), // bbox
                                                                 // .with_postive_point(223., 140.) // example 2
                                                                 // .with_postive_point(488., 523.), // example 2
                                                                 // .with_postive_point(221., 482.) // example 3
                                                                 // .with_postive_point(498., 633.) // example 3
                                                                 // .with_postive_point(750., 379.), // example 3
                                                                 // .with_bbox(310., 228., 424., 296.) // bbox example 7
                                                                 // .with_bbox(45., 260., 515., 470.), // bbox example 7
        ];
        let ys = model.run(&xs, &prompts)?;
        annotator.annotate(&xs, &ys);
    }

    Ok(())
}
