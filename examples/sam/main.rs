use anyhow::Result;
use usls::{
    models::{SamKind, SamPrompt, SAM},
    Annotator, DataLoader, Options, Scale,
};

#[derive(argh::FromArgs)]
/// Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// scale
    #[argh(option, default = "String::from(\"t\")")]
    scale: String,

    /// SAM kind
    #[argh(option, default = "String::from(\"sam\")")]
    kind: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();
    // Build model
    let (options_encoder, options_decoder) = match args.kind.as_str().try_into()? {
        SamKind::Sam => (
            Options::sam_v1_base_encoder(),
            Options::sam_v1_base_decoder(),
        ),
        SamKind::Sam2 => match args.scale.as_str().try_into()? {
            Scale::T => (Options::sam2_tiny_encoder(), Options::sam2_tiny_decoder()),
            Scale::S => (Options::sam2_small_encoder(), Options::sam2_small_decoder()),
            Scale::B => (
                Options::sam2_base_plus_encoder(),
                Options::sam2_base_plus_decoder(),
            ),
            _ => unimplemented!("Unsupported model scale: {:?}. Try b, s, t.", args.scale),
        },

        SamKind::MobileSam => (
            Options::mobile_sam_tiny_encoder(),
            Options::mobile_sam_tiny_decoder(),
        ),
        SamKind::SamHq => (
            Options::sam_hq_tiny_encoder(),
            Options::sam_hq_tiny_decoder(),
        ),
        SamKind::EdgeSam => (
            Options::edge_sam_3x_encoder(),
            Options::edge_sam_3x_decoder(),
        ),
    };

    let options_encoder = options_encoder
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let options_decoder = options_decoder.commit()?;
    let mut model = SAM::new(options_encoder, options_decoder)?;

    // Load image
    let xs = [DataLoader::try_read("images/truck.jpg")?];

    // Build annotator
    let annotator = Annotator::default().with_saveout(model.spec());

    // Prompt
    let prompts = vec![
        SamPrompt::default()
            // .with_postive_point(500., 375.), // postive point
            // .with_negative_point(774., 366.),   // negative point
            .with_bbox(215., 297., 643., 459.), // bbox
    ];

    // Run & Annotate
    let ys = model.forward(&xs, &prompts)?;
    annotator.annotate(&xs, &ys);

    Ok(())
}
