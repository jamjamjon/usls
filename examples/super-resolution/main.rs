use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{Swin2SR, APISR},
    Config, DataLoader, Model, Source,
};

mod apisr;
mod swin2sr;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Super-Resolution Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: images
    #[arg(long, global = true, default_value = "images/ekko.jpg")]
    pub source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Apisr(apisr::APISRArgs),
    Swin2sr(swin2sr::Swin2SRArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    match &cli.command {
        Commands::Apisr(args) => {
            let config = apisr::config(args)?.commit()?;
            run::<APISR>(config, &cli.source)
        }
        Commands::Swin2sr(args) => {
            let config = swin2sr::config(args)?.commit()?;
            run::<Swin2SR>(config, &cli.source)
        }
    }?;

    usls::perf(false);

    Ok(())
}

fn run<M>(config: Config, source: &Source) -> Result<()>
where
    for<'a> M: Model<Input<'a> = &'a [usls::Image]>,
{
    let mut model = M::new(config)?;

    // Build dataloader
    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    // Run & Save
    for xs in &dl {
        let ys = model.run(&xs)?;
        for y in ys {
            let images = y.images();
            if !images.is_empty() {
                for image in images.iter() {
                    image.save(format!(
                        "{}.png",
                        usls::Dir::Current
                            .base_dir_with_subs(&["runs/super-resolution", model.spec()])?
                            .join(usls::timestamp(None))
                            .display(),
                    ))?;
                }
            }
        }
    }

    Ok(())
}
