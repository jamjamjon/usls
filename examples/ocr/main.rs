use anyhow::Result;
use clap::{Parser, Subcommand};
use usls::{
    models::{Fast, LinkNet, PPDocLayout, PicoDet, SLANet, DB, SVTR, YOLO},
    vlm::TrOCR,
    Annotator, Config, DataLoader, Model, Source,
};

mod db;
mod doclayout_yolo;
mod fast;
mod linknet;
mod picodet_layout;
mod pp_doclayout;
mod slanet;
mod svtr;
mod trocr;
#[path = "../utils/mod.rs"]
mod utils;

const RUN_TEXT_DET: &str = "runs/text-detection";
const RUN_DOC_LAYOUT: &str = "runs/doc-layout";
const RUN_TABLE: &str = "runs/table-detection";

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
    PpDoclayout(pp_doclayout::PPDoclayoutArgs),
}

fn run<M>(config: Config, source: &Source, annotator: &Annotator, run_dir: &str) -> Result<()>
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
                        .base_dir_with_subs(&[run_dir, model.spec()])?
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

    match &cli.command {
        Commands::Db(args) => {
            let config = db::config(args)?.commit()?;
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
            run::<DB>(config, &cli.source, &annotator, RUN_TEXT_DET)
        }
        Commands::Fast(args) => {
            let config = fast::config(args)?.commit()?;
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
            run::<Fast>(config, &cli.source, &annotator, RUN_TEXT_DET)
        }
        Commands::DoclayoutYolo(args) => {
            let config = doclayout_yolo::config(args)?.commit()?;
            let annotator = Annotator::default();
            run::<YOLO>(config, &cli.source, &annotator, RUN_DOC_LAYOUT)
        }
        Commands::Linknet(args) => {
            let config = linknet::config(args)?.commit()?;
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
            run::<LinkNet>(config, &cli.source, &annotator, RUN_TEXT_DET)
        }
        Commands::PicodetLayout(args) => {
            let config = picodet_layout::config(args)?.commit()?;
            let annotator = Annotator::default();
            run::<PicoDet>(config, &cli.source, &annotator, RUN_DOC_LAYOUT)
        }
        Commands::Slanet(args) => {
            let config = slanet::config(args)?.commit()?;
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
            run::<SLANet>(config, &cli.source, &annotator, RUN_TABLE)
        }
        Commands::Svtr(args) => {
            let config = svtr::config(args)?.commit()?;
            run_svtr(config, &cli.source)
        }
        Commands::Trocr(args) => {
            let config = trocr::config(args)?.commit()?;
            run_trocr(config, &cli.source)
        }
        Commands::PpDoclayout(args) => {
            let config = pp_doclayout::config(args)?.commit()?;
            let annotator = Annotator::default()
                .with_hbb_style(
                    usls::HbbStyle::default()
                        .with_visible(false)
                        .with_text_visible(false),
                )
                .with_mask_style(
                    usls::MaskStyle::default()
                        .with_visible(false)
                        .with_cutout(true)
                        .with_draw_obbs(true), // .with_draw_polygon_largest(true),
                );
            run::<PPDocLayout>(config, &cli.source, &annotator, RUN_DOC_LAYOUT)
        }
    }?;

    usls::perf(false);
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
