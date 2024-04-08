use usls::{models::Clip, ops, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // visual
    let options_visual = Options::default()
        .with_model("../models/clip-b32-visual-dyn.onnx")
        .with_i00((1, 1, 4).into())
        .with_profile(false);

    // textual
    let options_textual = Options::default()
        .with_model("../models/clip-b32-textual-dyn.onnx")
        .with_tokenizer("tokenizer-clip.json")
        .with_i00((1, 1, 4).into())
        .with_profile(false);

    // build model
    let model = Clip::new(options_visual, options_textual)?;

    // texts
    let texts = vec![
        "A photo of a dinosaur ".to_string(),
        "A photo of a cat".to_string(),
        "A photo of a dog".to_string(),
        "几个胡萝卜".to_string(),
        "There are some playing cards on a striped table cloth".to_string(),
        "There is a doll with red hair and a clock on a table".to_string(),
        "Some people holding wine glasses in a restaurant".to_string(),
    ];
    let feats_text = model.encode_texts(&texts)?; // [n, ndim]

    // load image
    let dl = DataLoader::default()
        .with_batch(model.batch_visual())
        .load("./examples/clip/images")?;

    // loop
    for (images, paths) in dl {
        let feats_image = model.encode_images(&images).unwrap();

        // use image to query texts
        let matrix = ops::dot2(&feats_image, &feats_text)?; // [m, n]

        // summary
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
            println!("{:?}\n", probs);
        }
    }

    Ok(())
}
