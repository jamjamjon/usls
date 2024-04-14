use crate::GITHUB_ASSETS;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

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

pub const COCO_SKELETON_17: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];
pub const COCO_KEYPOINT_NAMES_17: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

pub const COCO_NAMES_80: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
