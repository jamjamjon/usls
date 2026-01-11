use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::Blip,
    vlm::{FastVLM, Florence2, Moondream2, SmolVLM},
    Annotator, Config, DataLoader, Model, Source, Task,
};

mod blip;
mod fastvlm;
mod florence2;
mod moondream2;
mod smolvlm;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "Vision-Language Model Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./assets/bus.jpg")]
    pub source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Blip(blip::BlipArgs),
    Fastvlm(fastvlm::FastvlmArgs),
    Florence2(florence2::Florence2Args),
    Moondream2(moondream2::Moondream2Args),
    Smolvlm(smolvlm::SmolvlmArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    match &cli.command {
        Commands::Blip(args) => {
            let config = blip::config(args)?.commit()?;
            run_blip(config, &cli.source, args)
        }
        Commands::Fastvlm(args) => {
            let config = fastvlm::config(args)?.commit()?;
            run_fastvlm(config, &cli.source, args)
        }
        Commands::Florence2(args) => {
            let config = florence2::config(args)?.commit()?;
            run_florence2(config, &cli.source, args)
        }
        Commands::Moondream2(args) => {
            let config = moondream2::config(args)?.commit()?;
            run_moondream2(config, &cli.source, args)
        }
        Commands::Smolvlm(args) => {
            let config = smolvlm::config(args)?.commit()?;
            run_smolvlm(config, &cli.source, args)
        }
    }?;

    usls::perf(false);
    Ok(())
}

fn run_blip(config: Config, source: &Source, args: &blip::BlipArgs) -> Result<()> {
    let mut model = Blip::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &xs {
        let ys = model.forward((&xs, args.prompt.as_deref()))?;
        println!("{:#?}", ys);
    }

    Ok(())
}

fn run_fastvlm(config: Config, source: &Source, args: &fastvlm::FastvlmArgs) -> Result<()> {
    let mut model = FastVLM::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &xs {
        let ys = model.forward((&xs, &args.prompt))?;
        for y in ys.iter() {
            let texts = y.texts();
            if !texts.is_empty() {
                for text in texts {
                    println!("\n[User]: {}\n[Assistant]: {:?}", args.prompt, text);
                }
            }
        }
    }

    Ok(())
}

fn run_florence2(config: Config, source: &Source, args: &florence2::Florence2Args) -> Result<()> {
    let mut model = Florence2::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &xs {
        let task: Task = args.task.parse()?;
        let ys = model.forward(&xs, &task)?;

        match task {
            Task::Caption(_)
            | Task::Ocr
            | Task::RegionToCategory(..)
            | Task::RegionToDescription(..) => {
                println!("Task: {:?}\n{:?}\n", task, &ys)
            }
            Task::DenseRegionCaption
            | Task::RegionProposal
            | Task::ObjectDetection
            | Task::OpenSetDetection(_)
            | Task::CaptionToPhraseGrounding(_)
            | Task::ReferringExpressionSegmentation(_)
            | Task::RegionToSegmentation(..)
            | Task::OcrWithRegion => {
                let annotator = Annotator::default()
                    .with_hbb_style(usls::HbbStyle::default().show_confidence(false));

                for (x, y) in xs.iter().zip(ys.iter()) {
                    if !y.is_empty() {
                        annotator.annotate(x, y)?.save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&["runs", "Florence2", model.decoder.spec()])?
                                .join(usls::timestamp(None))
                                .display()
                        ))?;
                    }
                }
            }
            _ => println!("{:?}", ys),
        }
    }

    Ok(())
}

fn run_moondream2(
    config: Config,
    source: &Source,
    args: &moondream2::Moondream2Args,
) -> Result<()> {
    let mut model = Moondream2::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &xs {
        let task: Task = args.task.parse()?;
        let ys = model.forward((&xs, &task))?;

        match task {
            Task::Caption(_) => {
                println!("{}:", task);
                for (i, y) in ys.iter().enumerate() {
                    let texts = y.texts();
                    if !texts.is_empty() {
                        println!("Image {}: {:?}\n", i, texts[0]);
                    }
                }
            }
            Task::Vqa(query) => {
                println!("Question: {}", query);
                for (i, y) in ys.iter().enumerate() {
                    let texts = y.texts();
                    if !texts.is_empty() {
                        println!("Image {}: {:?}\n", i, texts[0]);
                    }
                }
            }
            Task::OpenSetDetection(_) | Task::OpenSetKeypointsDetection(_) => {
                let annotator = Annotator::default()
                    .with_hbb_style(
                        usls::HbbStyle::default()
                            .with_draw_fill(true)
                            .show_confidence(false),
                    )
                    .with_keypoint_style(
                        usls::KeypointStyle::default()
                            .show_confidence(false)
                            .show_id(true)
                            .show_name(false),
                    );

                for (x, y) in xs.iter().zip(ys.iter()) {
                    if !y.is_empty() {
                        annotator.annotate(x, y)?.save(format!(
                            "{}.jpg",
                            usls::Dir::Current
                                .base_dir_with_subs(&["runs", "moondream2"])?
                                .join(usls::timestamp(None))
                                .display()
                        ))?;
                    }
                }
            }
            _ => println!("{:?}", ys),
        }
    }

    Ok(())
}

fn run_smolvlm(config: Config, source: &Source, args: &smolvlm::SmolvlmArgs) -> Result<()> {
    let mut model = SmolVLM::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &xs {
        let ys = model.forward((&xs, &args.prompt))?;
        for y in ys.iter() {
            let texts = y.texts();
            if !texts.is_empty() {
                for text in texts {
                    println!("[User]: {}\n\n[Assistant]: {:?}", args.prompt, text);
                }
            }
        }
    }

    Ok(())
}
