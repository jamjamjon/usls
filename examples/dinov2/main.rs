use usearch::ffi::{IndexOptions, MetricKind, ScalarKind};
use usls::{models::Dinov2, DataLoader, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build model
    let options = Options::default()
        .with_model("../models/dinov2-s14-dyn-f16.onnx")
        .with_i00((1, 1, 1).into())
        .with_i02((224, 224, 224).into())
        .with_i03((224, 224, 224).into());
    let mut model = Dinov2::new(&options)?;

    // build dataloader
    let dl = DataLoader::default()
        .with_batch(model.batch.opt as usize)
        .load("./examples/dinov2/images")?;

    // load query
    let query = image::io::Reader::open("./assets/bus.jpg")?.decode()?;
    let query = model.run(&[query])?;

    // build index
    let options = IndexOptions {
        dimensions: 384, // 768 for vitb; 384 for vits
        metric: MetricKind::L2sq,
        quantization: ScalarKind::F16,
        ..Default::default()
    };
    let index = usearch::new_index(&options)?;
    index.reserve(dl.clone().count())?;

    // load feats
    for (idx, (image, _path)) in dl.clone().enumerate() {
        let y = model.run(&image)?;
        index.add(idx as u64, &y.into_raw_vec())?;
    }

    // output
    let topk = 10;
    let matches = index.search(&query.into_raw_vec(), topk)?;
    let paths = dl.paths;
    for (idx, (k, score)) in matches
        .keys
        .into_iter()
        .zip(matches.distances.into_iter())
        .enumerate()
    {
        println!(
            "Top-{} distance: {:?} => {:?}",
            idx + 1,
            score,
            paths[k as usize]
        );
    }

    Ok(())
}
