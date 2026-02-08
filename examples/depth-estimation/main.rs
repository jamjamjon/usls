use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{DepthAnything, DepthPro},
    Annotator, Config, DataLoader, Model, Source,
};

mod depth_anything;
mod depth_pro;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser, Debug)]
#[command(author, version, about = "Depth-Estimation Example")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "images/street.jpg")]
    pub source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    DepthAnything(depth_anything::DepthAnythingArgs),
    DepthPro(depth_pro::DepthProArgs),
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
        // println!("{:?}", ys);
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/depth-estimation", model.spec()])?
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
    let annotator = Annotator::default().with_mask_style(
        usls::MaskStyle::default()
            .with_cutout(false)
            .with_colormap256("turbo".parse()?),
    );
    match &cli.command {
        Commands::DepthAnything(args) => {
            let config = depth_anything::config(args)?.commit()?;
            run::<DepthAnything>(config, &cli.source, &annotator)
        }
        Commands::DepthPro(args) => {
            let config = depth_pro::config(args)?.commit()?;
            run::<DepthPro>(config, &cli.source, &annotator)
        }
    }?;

    usls::perf_chart();

    Ok(())
}
