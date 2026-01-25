use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{GroundingDINO, OWLv2},
    Annotator, Config, DataLoader, Model, Source,
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
        default_value = "person,bus,a dog,cat,stop sign,tie,eye glasses,tree,camera,hand,a shoe,car,balcony,window"
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
    let annotator = Annotator::default();

    match &cli.command {
        Commands::GroundingDINO(args) => {
            let config = grounding_dino::config(args)?
                .with_class_confs(&cli.confs)
                .with_text_names_owned(cli.prompts)
                .commit()?;
            run::<GroundingDINO>(config, &cli.source, &annotator)
        }
        Commands::Owlv2(args) => {
            let config = owlv2::config(args)?
                .with_text_names_owned(cli.prompts)
                .commit()?;
            run::<OWLv2>(config, &cli.source, &annotator)
        }
    }?;
    usls::perf(false);

    Ok(())
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
                        .base_dir_with_subs(&["runs/open-set-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}
