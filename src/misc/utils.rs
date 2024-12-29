#![allow(dead_code)]

use indicatif::{ProgressBar, ProgressStyle};
use rand::{distributions::Alphanumeric, thread_rng, Rng};

pub(crate) const PREFIX_LENGTH: usize = 12;
pub(crate) const NETWORK_PREFIXES: &[&str] = &[
    "http://", "https://", "ftp://", "ftps://", "sftp://", "rtsp://", "mms://", "mmsh://",
    "rtmp://", "rtmps://", "file://",
];
pub(crate) const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"];
pub(crate) const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "mpeg", "mpg", "m4v", "m4p",
];
pub(crate) const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "aac", "ogg", "wma"];
pub(crate) const STREAM_PROTOCOLS: &[&str] = &[
    "rtsp://", "rtsps://", "rtspu://", "rtmp://", "rtmps://", "hls://", "http://", "https://",
];
pub(crate) const PROGRESS_BAR_STYLE_CYAN: &str =
    "{prefix:>12.cyan.bold} {msg} {human_pos}/{human_len} |{bar}| {elapsed_precise}";
pub(crate) const PROGRESS_BAR_STYLE_GREEN: &str =
    "{prefix:>12.green.bold} {msg} {human_pos}/{human_len} |{bar}| {elapsed_precise}";
pub(crate) const PROGRESS_BAR_STYLE_CYAN_2: &str =
    "{prefix:>12.cyan.bold} {human_pos}/{human_len} |{bar}| {msg}";
pub(crate) const PROGRESS_BAR_STYLE_CYAN_3: &str =
    "{prefix:>12.cyan.bold} |{bar}| {human_pos}/{human_len} {msg}";
pub(crate) const PROGRESS_BAR_STYLE_GREEN_2: &str =
    "{prefix:>12.green.bold} {human_pos}/{human_len} |{bar}| {elapsed_precise}";
pub(crate) const PROGRESS_BAR_STYLE_FINISH: &str =
    "{prefix:>12.green.bold} {msg} for {human_len} iterations in {elapsed}";
pub(crate) const PROGRESS_BAR_STYLE_FINISH_2: &str =
    "{prefix:>12.green.bold} {msg} x{human_len} in {elapsed}";
pub(crate) const PROGRESS_BAR_STYLE_FINISH_3: &str =
    "{prefix:>12.green.bold} {msg} ({binary_total_bytes}) in {elapsed}";
pub(crate) const PROGRESS_BAR_STYLE_FINISH_4: &str = "{prefix:>12.green.bold} {msg} in {elapsed}";

pub(crate) fn try_fetch_stem<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<String> {
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

pub fn human_bytes(size: f64, use_binary: bool) -> String {
    let units = if use_binary {
        ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
    } else {
        ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    };

    let mut size = size;
    let mut unit_index = 0;
    let k = if use_binary { 1024. } else { 1000. };

    while size >= k && unit_index < units.len() - 1 {
        size /= k;
        unit_index += 1;
    }

    format!("{:.2} {}", size, units[unit_index])
}

pub(crate) fn string_random(n: usize) -> String {
    thread_rng()
        .sample_iter(&Alphanumeric)
        .take(n)
        .map(char::from)
        .collect()
}

pub(crate) fn string_now(delimiter: &str) -> String {
    let t_now = chrono::Local::now();
    let fmt = format!(
        "%Y{}%m{}%d{}%H{}%M{}%S{}%f",
        delimiter, delimiter, delimiter, delimiter, delimiter, delimiter
    );
    t_now.format(&fmt).to_string()
}

pub(crate) fn build_progress_bar(
    n: u64,
    prefix: &str,
    msg: Option<&str>,
    style_temp: &str,
) -> anyhow::Result<ProgressBar> {
    let pb = ProgressBar::new(n);
    pb.set_style(ProgressStyle::with_template(style_temp)?.progress_chars("██ "));
    pb.set_prefix(format!("{:>PREFIX_LENGTH$}", prefix));
    pb.set_message(msg.unwrap_or_default().to_string());

    Ok(pb)
}
