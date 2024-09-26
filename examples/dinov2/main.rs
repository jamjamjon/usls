use usls::{models::Dinov2, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("dinov2/s-dyn.onnx")?
        .with_ixx(0, 2, 224.into())
        .with_ixx(0, 3, 224.into());
    let mut model = Dinov2::new(options)?;
    let x = [DataLoader::try_read("images/bus.jpg")?];
    let y = model.run(&x)?;
    println!("{y:?}");

    // TODO:
    // query from vector
    // let ys = model.query_from_vec(
    //     "./assets/bus.jpg",
    //     &[
    //         "./examples/dinov2/images/bus.jpg",
    //         "./examples/dinov2/images/1.jpg",
    //         "./examples/dinov2/images/2.jpg",
    //     ],
    //     Metric::L2,
    // )?;

    // or query from folder
    // let ys = model.query_from_folder("./assets/bus.jpg", "./examples/dinov2/images", Metric::IP)?;

    // results
    // for (i, y) in ys.iter().enumerate() {
    //     println!(
    //         "Top-{:<3}{:.7} {}",
    //         i + 1,
    //         y.1,
    //         y.2.canonicalize()?.display()
    //     );
    // }

    Ok(())
}
