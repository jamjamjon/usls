use anyhow::Result;
use usls::{models::PicoDet, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    // options
    let options = Options::picodet_layout_1x()
        // picodet_l_layout_3cls()
        // picodet_l_layout_17cls()
        .commit()?;
    let mut model = PicoDet::new(options)?;

    // load
    let xs = [DataLoader::try_read("images/academic.jpg")?];

    // annotator
    let annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout(model.spec());

    // run
    let ys = model.forward(&xs)?;
    println!("{:?}", ys);
    annotator.annotate(&xs, &ys);

    Ok(())
}
