#![allow(dead_code)]

use indicatif::{ProgressBar, ProgressStyle};
use rand::{distributions::Alphanumeric, thread_rng, Rng};

pub mod colormap256;
pub mod names;

pub use colormap256::*;
pub use names::*;

pub(crate) const CHECK_MARK: &str = "✅";
pub(crate) const CROSS_MARK: &str = "❌";
pub(crate) const SAFE_CROSS_MARK: &str = "❎";
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
    "{prefix:.cyan.bold} {msg} {human_pos}/{human_len} |{bar}| {elapsed_precise}";
pub(crate) const PROGRESS_BAR_STYLE_GREEN: &str =
    "{prefix:.green.bold} {msg} {human_pos}/{human_len} |{bar}| {elapsed_precise}";

pub fn human_bytes(size: f64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    let mut size = size;
    let mut unit_index = 0;
    let k = 1024.;

    while size >= k && unit_index < units.len() - 1 {
        size /= k;
        unit_index += 1;
    }

    format!("{:.1} {}", size, units[unit_index])
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

pub fn build_progress_bar(
    n: u64,
    prefix: &str,
    msg: Option<&str>,
    style_temp: &str,
) -> anyhow::Result<ProgressBar> {
    let pb = ProgressBar::new(n);
    pb.set_style(ProgressStyle::with_template(style_temp)?.progress_chars("██ "));
    pb.set_prefix(prefix.to_string());
    pb.set_message(msg.unwrap_or_default().to_string());

    Ok(pb)
}
