use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{DEIM, DFINE, RFDETR, RTDETR},
    Annotator, Config, DataLoader, Model, Source,
};

mod deim;
mod dfine;
mod rfdetr;
mod rtdetr;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Object Detection Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, value_delimiter = ',', default_values_t = vec![0.5])]
    pub confs: Vec<f32>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Rtdetr(rtdetr::RtdetrArgs),
    Rfdetr(rfdetr::RfdetrArgs),
    Deim(deim::DeimArgs),
    Dfine(dfine::DfineArgs),
}

fn run<M>(config: Config, source: &Source, annotator: &Annotator) -> Result<()>
where
    for<'a> M: Model<Input<'a> = &'a [usls::Image]>,
{
    let mut model = M::new(config)?;
    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &dl {
        let ys = model.forward(&xs)?;
        tracing::info!("{:?}", ys);
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/object-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    let annotator = Annotator::default();

    match &cli.command {
        Commands::Rfdetr(args) => {
            let config = rfdetr::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            run::<RFDETR>(config, &cli.source, &annotator)
        }
        Commands::Rtdetr(args) => {
            let config = rtdetr::config(args)?
                .with_class_confs(&cli.confs)
                .commit()?;
            run::<RTDETR>(config, &cli.source, &annotator)
        }
        Commands::Deim(args) => {
            let config = deim::config(args)?.with_class_confs(&cli.confs).commit()?;
            run::<DEIM>(config, &cli.source, &annotator)
        }
        Commands::Dfine(args) => {
            let config = dfine::config(args)?.with_class_confs(&cli.confs).commit()?;
            run::<DFINE>(config, &cli.source, &annotator)
        }
    }?;
    usls::perf(false);

    Ok(())
}
