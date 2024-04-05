use usls::{models::Dinov2, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/dinov2-s14-dyn-f16.onnx")
        // .with_model("../models/dinov2-b14-dyn.onnx")
        .with_i00((1, 1, 1).into())
        .with_i02((224, 224, 224).into())
        .with_i03((224, 224, 224).into());
    let _model = Dinov2::new(&options)?;
    println!("TODO...");

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
