pub(crate) const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"];
pub(crate) const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "mpeg", "mpg", "m4v", "m4p",
];
pub(crate) const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "aac", "ogg", "wma"];
pub(crate) const REMOTE_PROTOCOLS: &[&str] = &[
    "http://", "https://", "ftp://", "ftps://", "sftp://", "mms://", "mmsh://", "rtsp://",
    "rtmp://", "rtmps://", "file://",
];
pub(crate) const STREAM_PROTOCOLS: &[&str] = &[
    "rtsp://", "rtsps://", "rtspu://", "rtmp://", "rtmps://", "hls://",
];

/// Media location type indicating local or remote source.
#[derive(Debug, Clone, Default, Copy)]
pub enum Location {
    #[default]
    Local,
    Remote,
}

/// Stream type for media content.
#[derive(Debug, Clone, Copy, Default)]
pub enum StreamType {
    #[default]
    Pre,
    Live,
}

/// Media type classification for different content formats.
#[derive(Debug, Clone, Copy, Default)]
pub enum MediaType {
    #[default]
    Unknown,
    Image(Location),
    Video(Location),
    Audio(Location),
    Stream,
}

impl MediaType {
    pub fn is_possible_remote(s: &str) -> bool {
        // remote
        if REMOTE_PROTOCOLS.iter().any(|&p| s.starts_with(p)) {
            return true;
        }

        // local (in case of no network connection)
        if s.starts_with("./")
            || s.starts_with("../")
            || s.starts_with('/')
            || std::path::Path::new(s).exists()
        {
            return false;
        }

        // check out remote hub tags
        if s.split('/').collect::<Vec<&str>>().len() == 2 {
            let hub_tags = crate::Hub::default().tags();
            return hub_tags.iter().any(|tag| s.starts_with(tag));
        }

        // default
        false
    }

    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Self {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        if IMAGE_EXTENSIONS.contains(&extension.as_str()) {
            MediaType::Image(Location::Local)
        } else if VIDEO_EXTENSIONS.contains(&extension.as_str()) {
            MediaType::Video(Location::Local)
        } else if AUDIO_EXTENSIONS.contains(&extension.as_str()) {
            MediaType::Audio(Location::Local)
        } else {
            MediaType::Unknown
        }
    }

    pub fn from_url(url: &str) -> Self {
        if IMAGE_EXTENSIONS
            .iter()
            .any(|&ext| url.ends_with(&format!(".{}", ext)))
        {
            MediaType::Image(Location::Remote)
        } else if VIDEO_EXTENSIONS
            .iter()
            .any(|&ext| url.ends_with(&format!(".{}", ext)))
        {
            MediaType::Video(Location::Remote)
        } else if STREAM_PROTOCOLS
            .iter()
            .any(|&protocol| url.starts_with(protocol))
        {
            MediaType::Stream
        } else {
            MediaType::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_remote() {
        assert!(MediaType::is_possible_remote(
            "http://example.com/image.jpg"
        ));
        assert!(MediaType::is_possible_remote("rtsp://camera.local/stream"));
        // assert!(MediaType::is_possible_remote("images/image.jpg"));
        // assert!(MediaType::is_possible_remote("yolo/image.jpg"));
    }

    #[test]
    fn test_is_local() {
        assert!(MediaType::is_possible_remote(
            "http://example.com/image.jpg"
        )); // remote
        assert!(!MediaType::is_possible_remote("example.com/image.jpg"));
        assert!(!MediaType::is_possible_remote("./assets/bus.jpg"));
        assert!(!MediaType::is_possible_remote("assets/bus.jpg"));
        assert!(!MediaType::is_possible_remote("./images/image.jpg"));
        assert!(!MediaType::is_possible_remote("../images/image.jpg"));
        assert!(!MediaType::is_possible_remote("../../images/image.jpg"));
    }
}
