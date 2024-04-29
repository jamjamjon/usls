use usls::{models::SVTR, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_i00((1, 2, 8).into())
        .with_i03((320, 960, 1600).into())
        .with_confs(&[0.2])
        .with_vocab("ppocr_rec_vocab.txt")?
        .with_model("ppocr-v4-svtr-ch-dyn.onnx")?;
    let mut model = SVTR::new(&options)?;

    // load images
    let dl = DataLoader::default()
        .with_batch(1)
        .load("./examples/svtr/images")?;

    // run
    for (xs, paths) in dl {
        let ys = model.run(&xs)?;
        println!("{paths:?}: {:?}", ys[0].texts())
    }

    Ok(())
}
