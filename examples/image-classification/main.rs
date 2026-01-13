use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{BEiT, ConvNeXt, DeiT, FastViT, MobileOne, Ram},
    Annotator, Config, DataLoader, Model, Source,
};

mod beit;
mod convnext;
mod deit;
mod fastvit;
mod mobileone;
mod ram;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Image Classification Example", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[arg(long, global = true, default_value = "./assets/cat.png")]
    source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Mobileone(mobileone::MobileOneArgs),
    Deit(deit::DeiTArgs),
    Beit(beit::BEiTArgs),
    Convnext(convnext::ConvNextArgs),
    Fastvit(fastvit::FastViTArgs),
    Ram(ram::RamArgs),
}

fn main() -> anyhow::Result<()> {
    utils::init_logging();
    let cli = Cli::parse();
    let annotator = Annotator::default();

    match &cli.command {
        Commands::Mobileone(args) => {
            let config = mobileone::config(args)?.commit()?;
            run::<MobileOne>(config, &cli.source, &annotator)
        }
        Commands::Deit(args) => {
            let config = deit::config(args)?.commit()?;
            run::<DeiT>(config, &cli.source, &annotator)
        }
        Commands::Beit(args) => {
            let config = beit::config(args)?.commit()?;
            run::<BEiT>(config, &cli.source, &annotator)
        }
        Commands::Convnext(args) => {
            let config = convnext::config(args)?.commit()?;
            run::<ConvNeXt>(config, &cli.source, &annotator)
        }
        Commands::Fastvit(args) => {
            let config = fastvit::config(args)?.commit()?;
            run::<FastViT>(config, &cli.source, &annotator)
        }
        Commands::Ram(args) => {
            let config = ram::config(args)?.commit()?;
            run::<Ram>(config, &cli.source, &annotator)
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
        println!("{ys:?}");
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/image-classification", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}
