use usls::{models::SVTR, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_i00((1, 2, 8).into())
        .with_i03((320, 960, 1600).into())
        .with_confs(&[0.4])
        .with_vocab("../ppocr_rec_vocab.txt")
        .with_model("../models/ppocr-v4-svtr-ch-dyn.onnx");
    let mut model = SVTR::new(&options)?;

    // load image
    let xs = vec![
        DataLoader::try_read("./examples/svtr/text1.png")?,
        DataLoader::try_read("./examples/svtr/text2.png")?,
        DataLoader::try_read("./examples/svtr/text3.png")?,
        DataLoader::try_read("./examples/svtr/text4.png")?,
        DataLoader::try_read("./examples/svtr/text5.png")?,
    ];

    // run
    for text in model.run(&xs)?.into_iter() {
        println!("[Texts] {text}")
    }

    Ok(())
}
