mod device;
mod dtype;
mod dynconf;
mod iiix;
mod kind;
mod labels;
mod logits_sampler;
mod min_opt_max;
mod ops;
mod options;
mod processor;
mod retry;
mod scale;
mod task;
mod traits;
mod ts;
mod version;

pub use device::Device;
pub use dtype::DType;
pub use dynconf::DynConf;
pub(crate) use iiix::Iiix;
pub use kind::Kind;
pub use labels::*;
pub use logits_sampler::LogitsSampler;
pub use min_opt_max::MinOptMax;
pub use ops::*;
pub use options::*;
pub use processor::*;
pub use scale::Scale;
pub use task::Task;
pub use traits::*;
pub use ts::Ts;
pub use version::Version;

pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
pub const PREFIX_LENGTH: usize = 12;
pub const PROGRESS_BAR_STYLE_FINISH: &str =
    "{prefix:>12.green.bold} {msg} for {human_len} iterations in {elapsed}";
pub const PROGRESS_BAR_STYLE_FINISH_2: &str =
    "{prefix:>12.green.bold} {msg} x{human_len} in {elapsed}";
pub const PROGRESS_BAR_STYLE_FINISH_3: &str =
    "{prefix:>12.green.bold} {msg} ({binary_total_bytes}) in {elapsed}";
pub const PROGRESS_BAR_STYLE_CYAN_2: &str =
    "{prefix:>12.cyan.bold} {human_pos}/{human_len} |{bar}| {msg}";

pub fn build_resizer_filter(
    ty: &str,
) -> anyhow::Result<(fast_image_resize::Resizer, fast_image_resize::ResizeOptions)> {
    use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
    let ty = match ty.to_lowercase().as_str() {
        "box" => FilterType::Box,
        "bilinear" => FilterType::Bilinear,
        "hamming" => FilterType::Hamming,
        "catmullrom" => FilterType::CatmullRom,
        "mitchell" => FilterType::Mitchell,
        "gaussian" => FilterType::Gaussian,
        "lanczos3" => FilterType::Lanczos3,
        _ => anyhow::bail!("Unsupported resizer filter: {}", ty),
    };
    Ok((
        Resizer::new(),
        ResizeOptions::new().resize_alg(ResizeAlg::Convolution(ty)),
    ))
}

pub fn try_fetch_file_stem<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<String> {
    let p = p.as_ref();
    let stem = p
        .file_stem()
        .ok_or(anyhow::anyhow!(
            "Failed to get the `file_stem` of `model_file`: {:?}",
            p
        ))?
        .to_str()
        .ok_or(anyhow::anyhow!("Failed to convert from `&OsStr` to `&str`"))?;

    Ok(stem.to_string())
}

// TODO
pub fn build_progress_bar(
    n: u64,
    prefix: &str,
    msg: Option<&str>,
    style_temp: &str,
) -> anyhow::Result<indicatif::ProgressBar> {
    let pb = indicatif::ProgressBar::new(n);
    pb.set_style(indicatif::ProgressStyle::with_template(style_temp)?.progress_chars("██ "));
    pb.set_prefix(format!("{:>PREFIX_LENGTH$}", prefix));
    pb.set_message(msg.unwrap_or_default().to_string());

    Ok(pb)
}

pub fn human_bytes_decimal(size: f64, decimal_places: usize) -> String {
    const DECIMAL_UNITS: [&str; 7] = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    format_bytes_internal(size, 1000.0, &DECIMAL_UNITS, decimal_places)
}

pub fn human_bytes_binary(size: f64, decimal_places: usize) -> String {
    const BINARY_UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    format_bytes_internal(size, 1024.0, &BINARY_UNITS, decimal_places)
}

fn format_bytes_internal(
    mut size: f64,
    base: f64,
    units: &[&str],
    decimal_places: usize,
) -> String {
    let mut unit_index = 0;
    while size >= base && unit_index < units.len() - 1 {
        size /= base;
        unit_index += 1;
    }

    format!(
        "{:.precision$} {}",
        size,
        units[unit_index],
        precision = decimal_places
    )
}

pub fn generate_random_string(length: usize) -> String {
    use rand::{distributions::Alphanumeric, thread_rng, Rng};
    if length == 0 {
        return String::new();
    }
    let rng = thread_rng();
    let mut result = String::with_capacity(length);
    result.extend(rng.sample_iter(&Alphanumeric).take(length).map(char::from));
    result
}

pub fn timestamp(delimiter: Option<&str>) -> String {
    let delimiter = delimiter.unwrap_or("");
    let format = format!("%Y{0}%m{0}%d{0}%H{0}%M{0}%S{0}%f", delimiter);
    chrono::Local::now().format(&format).to_string()
}
