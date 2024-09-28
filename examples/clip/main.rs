use usls::{models::Clip, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // visual
    let options_visual = Options::default().with_model("clip/visual-base-dyn.onnx")?;

    // textual
    let options_textual = Options::default()
        .with_model("clip/textual-base-dyn.onnx")?
        .with_tokenizer("clip/tokenizer.json")?;

    // build model
    let mut model = Clip::new(options_visual, options_textual)?;

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
    let dl = DataLoader::new("./examples/clip/images")?.build()?;

    // loop
    for (images, paths) in dl {
        let feats_image = model.encode_images(&images).unwrap();

        // use image to query texts
        let matrix = match feats_image.embedding() {
            Some(x) => x.dot2(feats_text.embedding().unwrap())?,
            None => continue,
        };

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
