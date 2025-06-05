/// The name of the current crate.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// Standard prefix length for progress bar formatting.
pub const PREFIX_LENGTH: usize = 12;
/// Progress bar style for completion with iteration count.
pub const PROGRESS_BAR_STYLE_FINISH: &str =
    "{prefix:>12.green.bold} {msg} for {human_len} iterations in {elapsed}";
/// Progress bar style for completion with multiplier format.
pub const PROGRESS_BAR_STYLE_FINISH_2: &str =
    "{prefix:>12.green.bold} {msg} x{human_len} in {elapsed}";
/// Progress bar style for completion with byte size information.
pub const PROGRESS_BAR_STYLE_FINISH_3: &str =
    "{prefix:>12.green.bold} {msg} ({binary_total_bytes}) in {elapsed}";
/// Progress bar style for ongoing operations with position indicator.
pub const PROGRESS_BAR_STYLE_CYAN_2: &str =
    "{prefix:>12.cyan.bold} {human_pos}/{human_len} |{bar}| {msg}";

pub(crate) fn build_resizer_filter(
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

pub(crate) fn try_fetch_file_stem<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<String> {
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

pub(crate) fn build_progress_bar(
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

/// Formats a byte size into a human-readable string using decimal (base-1000) units.
///
/// # Arguments
/// * `size` - The size in bytes to format
/// * `decimal_places` - Number of decimal places to show in the formatted output
///
/// # Returns
/// A string representing the size with appropriate decimal unit (B, KB, MB, etc.)
///
/// # Example
/// ```ignore
/// let size = 1500000.0;
/// let formatted = human_bytes_decimal(size, 2);
/// assert_eq!(formatted, "1.50 MB");
/// ```
pub fn human_bytes_decimal(size: f64, decimal_places: usize) -> String {
    const DECIMAL_UNITS: [&str; 7] = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    format_bytes_internal(size, 1000.0, &DECIMAL_UNITS, decimal_places)
}

/// Formats a byte size into a human-readable string using binary (base-1024) units.
///
/// # Arguments
/// * `size` - The size in bytes to format
/// * `decimal_places` - Number of decimal places to show in the formatted output
///
/// # Returns
/// A string representing the size with appropriate binary unit (B, KiB, MiB, etc.)
///
/// # Example
/// ```ignore
/// let size = 1024.0 * 1024.0; // 1 MiB in bytes
/// let formatted = human_bytes_binary(size, 2);
/// assert_eq!(formatted, "1.00 MiB");
/// ```
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

/// Generates a random alphanumeric string of the specified length.
///
/// # Arguments
/// * `length` - The desired length of the random string
///
/// # Returns
/// A string containing random alphanumeric characters (A-Z, a-z, 0-9).
/// Returns an empty string if length is 0.
///
/// # Example
/// ```ignore
/// let random = generate_random_string(8);
/// assert_eq!(random.len(), 8);
/// // Each character in the string will be alphanumeric
/// assert!(random.chars().all(|c| c.is_ascii_alphanumeric()));
/// ```
pub fn generate_random_string(length: usize) -> String {
    use rand::{distr::Alphanumeric, rng, Rng};
    if length == 0 {
        return String::new();
    }
    let rng = rng();
    let mut result = String::with_capacity(length);
    result.extend(rng.sample_iter(&Alphanumeric).take(length).map(char::from));
    result
}

/// Generates a timestamp string in the format "YYYYMMDDHHmmSSffffff" with an optional delimiter.
///
/// # Arguments
/// * `delimiter` - Optional string to insert between each component of the timestamp.
///   If None or empty string, components will be joined without delimiter.
///
/// # Returns
/// A string containing the current timestamp with the specified format.
///
/// # Example
/// ```ignore
/// use chrono::TimeZone;
///
/// // Without delimiter
/// let ts = timestamp(None);
/// assert_eq!(ts.len(), 20); // YYYYMMDDHHmmSSffffff
///
/// // With delimiter
/// let ts = timestamp(Some("-"));
/// // Format: YYYY-MM-DD-HH-mm-SS-ffffff
/// assert_eq!(ts.split('-').count(), 7);
/// ```
pub fn timestamp(delimiter: Option<&str>) -> String {
    let delimiter = delimiter.unwrap_or("");
    let format = format!("%Y{0}%m{0}%d{0}%H{0}%M{0}%S{0}%f", delimiter);
    chrono::Local::now().format(&format).to_string()
}
