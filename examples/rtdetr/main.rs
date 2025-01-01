use anyhow::Result;
use usls::{models::RTDETR, Annotator, DataLoader, Options};

fn main() -> Result<()> {
    // options
    let options = Options::rtdetr_v2_s_coco()
        // rtdetr_v1_r18vd_coco()
        // rtdetr_v2_ms_coco()
        // rtdetr_v2_m_coco()
        // rtdetr_v2_l_coco()
        // rtdetr_v2_x_coco()
        .commit()?;
    let mut model = RTDETR::new(options)?;

    // load
    let xs = [DataLoader::try_read("./assets/bus.jpg")?];

    // run
    let ys = model.forward(&xs)?;

    // extract bboxes
    for y in ys.iter() {
        if let Some(bboxes) = y.bboxes() {
            println!("[Bboxes]: Found {} objects", bboxes.len());
            for (i, bbox) in bboxes.iter().enumerate() {
                println!("{}: {:?}", i, bbox)
            }
        }
    }

    // annotate
    let annotator = Annotator::default()
        .with_bboxes_thickness(3)
        .with_saveout(model.spec());
    annotator.annotate(&xs, &ys);

    Ok(())
}
