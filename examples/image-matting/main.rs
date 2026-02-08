use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{BiRefNet, MODNet, MediaPipeSegmenter},
    Annotator, Config, DataLoader, Model, Source,
};

#[path = "../birefnet/args.rs"]
mod birefnet;
mod mediapipe;
mod modnet;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Image Matting Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "images/liuyifei.png")]
    pub source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Mediapipe(mediapipe::MediapipeArgs),
    Modnet(modnet::ModnetArgs),
    Birefnet(birefnet::BiRefNetArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    let annotator =
        Annotator::default().with_mask_style(usls::MaskStyle::default().with_cutout(false));

    match &cli.command {
        Commands::Mediapipe(args) => {
            let config = mediapipe::config(args)?.commit()?;
            run::<MediaPipeSegmenter>(config, &cli.source, &annotator)
        }
        Commands::Modnet(args) => {
            let config = modnet::config(args)?.commit()?;
            run::<MODNet>(config, &cli.source, &annotator)
        }
        Commands::Birefnet(args) => {
            let config = birefnet::config(args)?.commit()?;
            run::<BiRefNet>(config, &cli.source, &annotator)
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
        // println!("{:?}", ys);
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/image-matting", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}
