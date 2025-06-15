use anyhow::Result;
use ndarray::Axis;
use usls::{models::Clip, Config, DataLoader};

#[derive(argh::FromArgs)]
/// CLIP Example
struct Args {
    /// device
    #[argh(option, default = "String::from(\"cpu:0\")")]
    device: String,

    /// dtype
    #[argh(option, default = "String::from(\"fp16\")")]
    dtype: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();
    let args: Args = argh::from_env();

    // build model
    let config = Config::mobileclip_s0()
        // mobileclip_blt()
        // clip_vit_b16()
        // clip_vit_l14()
        // clip_vit_b32()
        // jina_clip_v1()
        // jina_clip_v2()
        .with_dtype_all(args.dtype.parse()?)
        .with_device_all(args.device.parse()?)
        .commit()?;
    let mut model = Clip::new(config)?;

    // texts
    let texts = vec![
        "A photo of a dinosaur.",
        "A photo of a cat.",
        "A photo of a dog.",
        "A picture of some carrots.",
        "There are some playing cards on a striped table cloth.",
        "There is a doll with red hair and a clock on a table.",
        "Some people holding wine glasses in a restaurant.",
    ];
    let feats_text = model.encode_texts(&texts)?.norm(1)?;

    // load images
    let dl = DataLoader::new("./examples/clip/images")?.build()?;

    // run
    for images in &dl {
        let feats_image = model.encode_images(&images)?.norm(1)?;

        // use image to query texts
        let matrix = (feats_image * 100.).dot2(&feats_text)?.softmax(1)?;

        // Process each image's matching scores
        for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
            let (id, &score) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            println!(
                "[{:.6}%] ({}) <=> ({})",
                score * 100.0,
                images[i].source().unwrap().display(),
                &texts[id]
            );
        }
    }
    usls::perf(false);

    Ok(())
}
