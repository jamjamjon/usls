use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{GroundingDINO, OWLv2},
    Annotator, DataLoader, Model, Source,
};

mod grounding_dino;
mod owlv2;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Open-Set Detection Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    /// Confidence thresholds (comma-separated for per-class, or single value for all)
    #[arg(long, global = true, value_delimiter = ',', default_values_t = vec![0.5])]
    pub confs: Vec<f32>,

    /// Prompts, labels
    #[arg(
        short = 'p',
        long,
        global = true,
        value_delimiter = ',',
        default_value = "person,bus,a dog,cat,stop sign,tie,eye glasses,tree,camera,hand,a shoe,balcony,window"
    )]
    pub prompts: Vec<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    GroundingDINO(grounding_dino::GroundingDINOArgs),
    Owlv2(owlv2::Owlv2Args),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    // Build dataloader
    let xs = DataLoader::new(&cli.source)?.try_read()?;

    // Build annotator
    let annotator = Annotator::default();

    match &cli.command {
        Commands::GroundingDINO(args) => {
            // Build model
            let config = grounding_dino::config(args)?
                .with_class_confs(&cli.confs)
                .with_text_names_owned(cli.prompts)
                .commit()?;
            let mut model = GroundingDINO::new(config)?;

            // Run & Annotate
            let ys = model.run(&xs)?;
            println!("{ys:?}");
            for (x, y) in xs.iter().zip(ys.iter()) {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/open-set-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display(),
                ))?;
            }
        }
        Commands::Owlv2(args) => {
            // Build model
            let config = owlv2::config(args)?
                .with_text_names_owned(cli.prompts)
                .commit()?;
            let mut model = OWLv2::new(config)?;

            // Run & Annotate
            let ys = model.run(&xs)?;
            println!("{ys:?}");
            for (x, y) in xs.iter().zip(ys.iter()) {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/open-set-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display(),
                ))?;
            }
        }
    }
    usls::perf(false);

    Ok(())
}
