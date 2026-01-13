use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{Fast, LinkNet, PicoDet, SLANet, DB, SVTR, YOLO},
    vlm::TrOCR,
    Annotator, Config, DataLoader, Model, Source,
};

mod db;
mod doclayout_yolo;
mod fast;
mod linknet;
mod picodet_layout;
mod slanet;
mod svtr;
mod trocr;
#[path = "../utils/mod.rs"]
mod utils;

#[derive(Parser)]
#[command(author, version, about = "OCR Examples")]
#[command(propagate_version = true)]
struct Cli {
    /// Source: image path, folder, or video
    #[arg(long, global = true, default_value = "./examples/ocr/images-det")]
    pub source: Source,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Db(db::DbArgs),
    Fast(fast::FastArgs),
    DoclayoutYolo(doclayout_yolo::DoclayoutYoloArgs),
    Linknet(linknet::LinknetArgs),
    PicodetLayout(picodet_layout::PicodetLayoutArgs),
    Slanet(slanet::SlanetArgs),
    Svtr(svtr::SvtrArgs),
    Trocr(trocr::TrocrArgs),
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    match &cli.command {
        Commands::Db(args) => {
            let config = db::config(args)?.commit()?;
            run_db(config, &cli.source, args)
        }
        Commands::Fast(args) => {
            let config = fast::config(args)?.commit()?;
            run_fast(config, &cli.source, args)
        }
        Commands::DoclayoutYolo(args) => {
            let config = doclayout_yolo::config(args)?.commit()?;
            run_doclayout_yolo(config, &cli.source)
        }
        Commands::Linknet(args) => {
            let config = linknet::config(args)?.commit()?;
            run_linknet(config, &cli.source, args)
        }
        Commands::PicodetLayout(args) => {
            let config = picodet_layout::config(args)?.commit()?;
            run_picodet_layout(config, &cli.source)
        }
        Commands::Slanet(args) => {
            let config = slanet::config(args)?.commit()?;
            run_slanet(config, &cli.source)
        }
        Commands::Svtr(args) => {
            let config = svtr::config(args)?.commit()?;
            run_svtr(config, &cli.source)
        }
        Commands::Trocr(args) => {
            let config = trocr::config(args)?.commit()?;
            run_trocr(config, &cli.source)
        }
    }?;

    usls::perf(false);
    Ok(())
}

fn run_db(config: Config, source: &Source, args: &db::DbArgs) -> Result<()> {
    let mut model = DB::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    let annotator = Annotator::default()
        .with_polygon_style(
            usls::PolygonStyle::default()
                .with_visible(true)
                .with_text_visible(true)
                .with_thickness(2)
                .show_confidence(args.show_polygons_conf)
                .show_id(false)
                .show_name(false)
                .with_outline_color(usls::ColorSource::Custom([255, 105, 180, 255].into())),
        )
        .with_hbb_style(
            usls::HbbStyle::default()
                .with_visible(args.show_hbbs)
                .with_text_visible(true)
                .with_thickness(1)
                .show_confidence(args.show_hbbs_conf)
                .show_id(false)
                .show_name(false),
        )
        .with_obb_style(
            usls::ObbStyle::default()
                .with_visible(args.show_obbs)
                .with_text_visible(true)
                .show_confidence(args.show_obbs_conf)
                .show_id(false)
                .show_name(false),
        );

    for xs in &xs {
        let ys = model.forward(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/text-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn run_fast(config: Config, source: &Source, args: &fast::FastArgs) -> Result<()> {
    let mut model = Fast::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    let annotator = Annotator::default()
        .with_polygon_style(
            usls::PolygonStyle::default()
                .with_visible(true)
                .with_text_visible(false)
                .show_confidence(args.show_polygons_conf)
                .show_id(args.show_polygons_id)
                .show_name(args.show_polygons_name)
                .with_outline_color(usls::ColorSource::Custom([255, 105, 180, 255].into())),
        )
        .with_hbb_style(
            usls::HbbStyle::default()
                .with_visible(false)
                .with_text_visible(false)
                .with_thickness(1)
                .show_confidence(false)
                .show_id(false)
                .show_name(false),
        )
        .with_obb_style(
            usls::ObbStyle::default()
                .with_visible(false)
                .with_text_visible(false)
                .show_confidence(false)
                .show_id(false)
                .show_name(false),
        );

    for xs in &xs {
        let ys = model.forward(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/text-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn run_doclayout_yolo(config: Config, source: &Source) -> Result<()> {
    let mut model = YOLO::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    let annotator = Annotator::default();

    for xs in &xs {
        let ys = model.forward(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/doc-layout", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn run_linknet(config: Config, source: &Source, args: &linknet::LinknetArgs) -> Result<()> {
    let mut model = LinkNet::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    let annotator = Annotator::default()
        .with_polygon_style(
            usls::PolygonStyle::default()
                .with_visible(true)
                .with_text_visible(false)
                .show_confidence(args.show_polygons_conf)
                .with_thickness(2)
                .show_id(args.show_polygons_id)
                .show_name(args.show_polygons_name)
                .with_outline_color(usls::ColorSource::Custom([255, 105, 180, 255].into())),
        )
        .with_hbb_style(
            usls::HbbStyle::default()
                .with_visible(false)
                .with_text_visible(false)
                .with_thickness(2)
                .show_confidence(false)
                .show_id(false)
                .show_name(false),
        )
        .with_obb_style(
            usls::ObbStyle::default()
                .with_visible(false)
                .with_text_visible(false)
                .show_confidence(false)
                .show_id(false)
                .show_name(false),
        );

    for xs in &xs {
        let ys = model.forward(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/text-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn run_picodet_layout(config: Config, source: &Source) -> Result<()> {
    let mut model = PicoDet::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    let annotator = Annotator::default();

    for xs in &xs {
        let ys = model.forward(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/doc-layout", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn run_slanet(config: Config, source: &Source) -> Result<()> {
    let mut model = SLANet::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    let annotator = Annotator::default().with_keypoint_style(
        usls::KeypointStyle::default()
            .with_text_visible(false)
            .with_skeleton(
                (
                    [(0, 1), (1, 2), (2, 3), (3, 0)],
                    [
                        usls::Color::black(),
                        usls::Color::red(),
                        usls::Color::green(),
                        usls::Color::blue(),
                    ],
                )
                    .into(),
            ),
    );

    for xs in &xs {
        let ys = model.forward(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            if !y.is_empty() {
                annotator.annotate(x, y)?.save(format!(
                    "{}.jpg",
                    usls::Dir::Current
                        .base_dir_with_subs(&["runs/table-detection", model.spec()])?
                        .join(usls::timestamp(None))
                        .display()
                ))?;
            }
        }
    }
    Ok(())
}

fn run_svtr(config: Config, source: &Source) -> Result<()> {
    let mut model = SVTR::new(config)?;
    let dl = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &dl {
        let ys = model.forward(&xs)?;
        println!("ys: {ys:?}");
    }
    Ok(())
}

fn run_trocr(config: Config, source: &Source) -> Result<()> {
    let mut model = TrOCR::new(config)?;
    let xs = DataLoader::new(source)?
        .with_batch(model.batch() as _)
        .with_progress_bar(true)
        .stream()?;

    for xs in &xs {
        let ys = model.forward(&xs)?;
        println!("{ys:?}");
    }
    Ok(())
}
