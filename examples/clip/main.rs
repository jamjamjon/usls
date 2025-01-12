use anyhow::Result;
use usls::{models::Clip, DataLoader, Ops, Options};

#[derive(argh::FromArgs)]
/// CLIP Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    let args: Args = argh::from_env();
    // build model
    let options_visual = Options::jina_clip_v1_visual()
        // clip_vit_b32_visual()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let options_textual = Options::jina_clip_v1_textual()
        // clip_vit_b32_textual()
        .with_model_device(args.device.as_str().try_into()?)
        .commit()?;
    let mut model = Clip::new(options_visual, options_textual)?;

    // texts
    let texts = vec![
        "A photo of a dinosaur",
        "A photo of a cat",
        "A photo of a dog",
        "Some carrots",
        "There are some playing cards on a striped table cloth",
        "There is a doll with red hair and a clock on a table",
        "Some people holding wine glasses in a restaurant",
    ];
    let feats_text = model.encode_texts(&texts)?; // [n, ndim]

    // load images
    let dl = DataLoader::new("./examples/clip/images")?.build()?;

    // run
    for (images, paths) in dl {
        let feats_image = model.encode_images(&images)?;

        // use image to query texts
        let matrix = Ops::dot2(&feats_image, &feats_text)?;

        for i in 0..paths.len() {
            let probs = &matrix[i];
            let (id, &score) = probs
                .iter()
                .enumerate()
                .reduce(|max, x| if x.1 > max.1 { x } else { max })
                .unwrap();

            println!(
                "({:?}%) {} => {} ",
                score * 100.0,
                paths[i].display(),
                &texts[id]
            );
        }
    }

    Ok(())
}
