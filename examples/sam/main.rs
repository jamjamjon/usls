use clap::Parser;

use usls::{
    models::{SamKind, SamPrompt, SAM},
    Annotator, DataLoader, Options,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, value_enum, default_value_t = SamKind::Sam)]
    pub kind: SamKind,

    #[arg(long, default_value_t = 0)]
    pub device_id: usize,

    #[arg(long)]
    pub use_low_res_mask: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Options
    let (options_encoder, options_decoder, saveout) = match args.kind {
        SamKind::Sam => {
            let options_encoder = Options::default()
                // .with_model("sam/sam-vit-b-encoder.onnx")?;
                .with_model("sam/sam-vit-b-encoder-u8.onnx")?;

            let options_decoder = Options::default()
                .with_sam_kind(SamKind::Sam)
                // .with_model("sam/sam-vit-b-decoder.onnx")?;
                // .with_model("sam/sam-vit-b-decoder-singlemask.onnx")?;
                .with_model("sam/sam-vit-b-decoder-u8.onnx")?;
            (options_encoder, options_decoder, "SAM")
        }
        SamKind::Sam2 => {
            let options_encoder = Options::default()
                // .with_model("sam/sam2-hiera-tiny-encoder.onnx")?;
                // .with_model("sam/sam2-hiera-small-encoder.onnx")?;
                .with_model("sam/sam2-hiera-base-plus-encoder.onnx")?;
            let options_decoder = Options::default()
                .with_sam_kind(SamKind::Sam2)
                // .with_model("sam/sam2-hiera-tiny-decoder.onnx")?;
                // .with_model("sam/sam2-hiera-small-decoder.onnx")?;
                .with_model("sam/sam2-hiera-base-plus-decoder.onnx")?;
            (options_encoder, options_decoder, "SAM2")
        }
        SamKind::MobileSam => {
            let options_encoder =
                Options::default().with_model("sam/mobile-sam-vit-t-encoder.onnx")?;

            let options_decoder = Options::default()
                .with_sam_kind(SamKind::MobileSam)
                .with_model("sam/mobile-sam-vit-t-decoder.onnx")?;
            (options_encoder, options_decoder, "Mobile-SAM")
        }
        SamKind::SamHq => {
            let options_encoder = Options::default().with_model("sam/sam-hq-vit-t-encoder.onnx")?;

            let options_decoder = Options::default()
                .with_sam_kind(SamKind::SamHq)
                .with_model("sam/sam-hq-vit-t-decoder.onnx")?;
            (options_encoder, options_decoder, "SAM-HQ")
        }
        SamKind::EdgeSam => {
            let options_encoder = Options::default().with_model("sam/edge-sam-3x-encoder.onnx")?;
            let options_decoder = Options::default()
                .with_sam_kind(SamKind::EdgeSam)
                .with_model("sam/edge-sam-3x-decoder.onnx")?;
            (options_encoder, options_decoder, "Edge-SAM")
        }
    };
    let options_encoder = options_encoder
        .with_cuda(args.device_id)
        .with_ixx(0, 2, (800, 1024, 1024).into())
        .with_ixx(0, 3, (800, 1024, 1024).into());
    let options_decoder = options_decoder
        .with_cuda(args.device_id)
        .use_low_res_mask(args.use_low_res_mask)
        .with_find_contours(true);

    // Build model
    let mut model = SAM::new(options_encoder, options_decoder)?;

    // Load image
    let xs = [
        DataLoader::try_read("images/truck.jpg")?,
        // DataLoader::try_read("images/dog.jpg")?,
    ];

    // Build annotator
    let annotator = Annotator::default().with_saveout(saveout);

    // Prompt
    let prompts = vec![
        SamPrompt::default()
            // .with_postive_point(500., 375.), // postive point
            // .with_negative_point(774., 366.),   // negative point
            .with_bbox(215., 297., 643., 459.), // bbox
    ];

    // Run & Annotate
    let ys = model.run(&xs, &prompts)?;
    annotator.annotate(&xs, &ys);

    Ok(())
}
