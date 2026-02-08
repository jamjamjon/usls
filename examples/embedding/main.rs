use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{Clip, DINO},
    Config, DataLoader, Model, Source,
};

mod clip;
mod dino;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Embedding Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./examples/embedding/images")]
    pub source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Clip(clip::ClipArgs),
    Dino(dino::DinoArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    match &cli.command {
        Commands::Clip(args) => {
            let config = clip::config(args)?.commit()?;
            run_clip(config, &cli.source, args)
        }
        Commands::Dino(args) => {
            let config = dino::config(args)?.commit()?;
            run_dino(config, &cli.source)
        }
    }?;

    usls::perf_chart();
    Ok(())
}

fn run_clip(config: Config, source: &Source, _args: &clip::ClipArgs) -> Result<()> {
    let mut model = Clip::new(config)?;

    let texts = vec![
        "A photo of a dinosaur.",
        "A photo of a cat.",
        "A photo of a dog.",
        "A picture of some carrots.",
        "There are some playing cards on a striped table cloth.",
        "There is a doll with red hair and a clock on a table.",
        "Some people holding wine glasses in a restaurant.",
    ];

    let feats_text = model.encode_texts(&texts)?.embedding;
    let feats_text_norm = feats_text.norm_l2_keepdim(-1)?;
    let feats_text = (feats_text / feats_text_norm).t()?;

    let dl = DataLoader::new(source)?.stream()?;

    for images in &dl {
        let feats_image = model.encode_images(&images)?.embedding;
        let feats_image_norm = feats_image.norm_l2_keepdim(-1)?;
        let feats_image = feats_image / feats_image_norm;

        let matrix = (feats_image * 100.0f32).matmul(&feats_text)?.softmax(1)?;

        for (i, row) in matrix.iter_dim(0).into_iter().enumerate() {
            let (id, &score) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            println!(
                "[{:.6}%] ({}) <=> ({})",
                score * 100.0,
                images[i].source().unwrap().display(),
                &texts[id]
            );
        }
    }

    Ok(())
}

fn run_dino(config: Config, source: &Source) -> Result<()> {
    let mut model = DINO::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &xs {
        let feats = model.encode_images(&xs)?;
        println!("Feat shape: {:?}", feats.embedding.shape());
    }

    Ok(())
}
