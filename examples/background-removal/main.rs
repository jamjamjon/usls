use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{BEN2, RMBG},
    Annotator, Config, DataLoader, Model, Source,
};

mod ben2;
mod rmbg;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Background Removal Example", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[arg(long, global = true, default_value = "./assets/cat.png")]
    source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Ben2(ben2::Ben2Args),
    Rmbg(rmbg::RmbgArgs),
}

fn main() -> anyhow::Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    let annotator =
        Annotator::default().with_mask_style(usls::MaskStyle::default().with_cutout(true));

    match &cli.command {
        Commands::Ben2(args) => {
            let config = ben2::config(args)?.commit()?;
            run::<BEN2>(config, &cli.source, &annotator)
        }
        Commands::Rmbg(args) => {
            let config = rmbg::config(args)?.commit()?;
            run::<RMBG>(config, &cli.source, &annotator)
        }
    }?;

    usls::perf_chart();

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
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/background-removal", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}
