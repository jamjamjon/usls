use anyhow::{anyhow, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{distributions::Alphanumeric, thread_rng, Rng};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub mod colormap256;
pub mod names;

pub use colormap256::*;
pub use names::*;

pub(crate) const GITHUB_ASSETS: &str =
    "https://github.com/jamjamjon/assets/releases/download/v0.0.1";
pub(crate) const CHECK_MARK: &str = "✅";
pub(crate) const CROSS_MARK: &str = "❌";
pub(crate) const SAFE_CROSS_MARK: &str = "❎";

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

pub(crate) fn auto_load<P: AsRef<Path>>(src: P, sub: Option<&str>) -> Result<String> {
    let src = src.as_ref();
    let p = if src.is_file() {
        src.into()
    } else {
        let sth = src.file_name().unwrap().to_str().unwrap();
        let mut p = home_dir(sub);
        p.push(sth);
        if !p.is_file() {
            download(
                &format!("{}/{}", GITHUB_ASSETS, sth),
                &p,
                Some(sth.to_string().as_str()),
            )?;
        }
        p
    };
    Ok(p.to_str().unwrap().to_string())
}

/// `download` sth from src to dst
pub fn download<P: AsRef<Path> + std::fmt::Debug>(
    src: &str,
    dst: P,
    prompt: Option<&str>,
) -> Result<()> {
    let resp = ureq::AgentBuilder::new()
        .try_proxy_from_env(true)
        .build()
        .get(src)
        .timeout(std::time::Duration::from_secs(2000))
        .call()
        .map_err(|err| anyhow!("Failed to download. {err:?}"))?;
    let ntotal = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .expect("Content-Length header should be present on archive response");
    let pb = ProgressBar::new(ntotal);
    pb.set_style(
            ProgressStyle::with_template(
                "{prefix:.bold} {msg:.dim} [{bar:.blue.bright/white.dim}] {binary_bytes}/{binary_total_bytes} ({binary_bytes_per_sec}, {percent_precise}%, {elapsed})"
            )
            .unwrap()
            .progress_chars("#>-"));
    pb.set_prefix(String::from("\n🐢 Downloading"));
    pb.set_message(prompt.unwrap_or_default().to_string());
    let mut reader = resp.into_reader();
    let mut buffer = [0; 256];
    let mut downloaded_bytes = 0usize;
    let mut f = std::fs::File::create(&dst).expect("Failed to create file");
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        pb.inc(bytes_read as u64);
        f.write_all(&buffer[..bytes_read])?;
        downloaded_bytes += bytes_read;
    }
    assert_eq!(downloaded_bytes as u64, ntotal);
    pb.finish();
    println!();
    Ok(())
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

#[allow(dead_code)]
pub(crate) fn config_dir() -> PathBuf {
    match dirs::config_dir() {
        Some(mut d) => {
            d.push("usls");
            if !d.exists() {
                std::fs::create_dir_all(&d).expect("Failed to create usls config directory.");
            }
            d
        }
        None => panic!("Unsupported operating system. Now support Linux, MacOS, Windows."),
    }
}

#[allow(dead_code)]
pub(crate) fn home_dir(sub: Option<&str>) -> PathBuf {
    match dirs::home_dir() {
        Some(mut d) => {
            d.push(".usls");
            if let Some(sub) = sub {
                d.push(sub);
            }
            if !d.exists() {
                std::fs::create_dir_all(&d).expect("Failed to create usls home directory.");
            }
            d
        }
        None => panic!("Unsupported operating system. Now support Linux, MacOS, Windows."),
    }
}
