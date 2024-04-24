use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub mod coco;
mod turbo;

pub use turbo::TURBO;

pub const GITHUB_ASSETS: &str = "https://github.com/jamjamjon/assets/releases/download/v0.0.1";
pub const CHECK_MARK: &str = "✅";
pub const CROSS_MARK: &str = "❌";
pub const SAFE_CROSS_MARK: &str = "❎";

pub fn auto_load<P: AsRef<Path>>(src: P) -> Result<String> {
    let src = src.as_ref();
    let p = if src.is_file() {
        src.into()
    } else {
        let sth = src.file_name().unwrap().to_str().unwrap();
        let mut p = config_dir();
        p.push(sth);
        // download from github assets if not exists in config directory
        if !p.is_file() {
            download(
                &format!("{}/{}", GITHUB_ASSETS, sth),
                &p,
                Some(sth.to_string().as_str()),
            )
            .unwrap_or_else(|err| panic!("Fail to load {:?}: {err}", src));
        }
        p
    };
    Ok(p.to_str().unwrap().to_string())
}

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
        .unwrap_or_else(|err| panic!("Failed to GET: {}", err));
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

pub fn string_now(delimiter: &str) -> String {
    let t_now = chrono::Local::now();
    let fmt = format!(
        "%Y{}%m{}%d{}%H{}%M{}%S{}%f",
        delimiter, delimiter, delimiter, delimiter, delimiter, delimiter
    );
    t_now.format(&fmt).to_string()
}

pub fn config_dir() -> PathBuf {
    match dirs::config_dir() {
        Some(mut d) => {
            d.push("usls");
            if !d.exists() {
                std::fs::create_dir_all(&d).expect("Failed to create config directory.");
            }
            d
        }
        None => panic!("Unsupported operating system. Now support Linux, MacOS, Windows."),
    }
}
