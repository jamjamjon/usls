use anyhow::Result;
use clap::Parser;
use usls::{models::Sapiens, Annotator, Config, DType, DataLoader, Device, Model};

#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Sapiens Example", long_about = None)]
#[command(propagate_version = true)]
struct Args {
    /// Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, default_value = "cpu")]
    pub device: Device,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

    /// DType: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "q4f16")]
    pub dtype: DType,

    /// Source: image, folder
    #[arg(long, default_value = "images/paul-george.jpg", value_delimiter = ',')]
    source: Vec<String>,

    /// num dry run
    #[arg(long, global = true, default_value_t = 0)]
    pub num_dry_run: usize,
}

fn main() -> Result<()> {
    utils::init_logging();
    let args = Args::parse();

    // build
    let config = Config::sapiens_seg_0_3b()
        .with_model_dtype(args.dtype)
        .with_model_device(args.device)
        .with_model_num_dry_run(args.num_dry_run)
        .with_image_processor_device(args.processor_device);

    let config = config.commit()?;
    let mut model = Sapiens::new(config)?;

    // load
    let xs = DataLoader::new(&args.source)?.try_read()?;

    // run
    let ys = model.run(&xs)?;

    // annotate
    let annotator = Annotator::default();
    for (x, y) in xs.iter().zip(ys.iter()) {
        annotator.annotate(x, y)?.save(format!(
            "{}.jpg",
            usls::Dir::Current
                .base_dir_with_subs(&["runs", model.spec()])?
                .join(usls::timestamp(None))
                .display(),
        ))?;
    }
    usls::perf(false);

    Ok(())
}
