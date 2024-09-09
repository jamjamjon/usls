use crate::{AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, STREAM_PROTOCOLS, VIDEO_EXTENSIONS};

// ----------------------------------
//     try this ???
// ----------------------------------
#[derive(Debug, Clone)]
pub enum MediaType {
    Image(Location),
    Video(Location),
    Audio(Location),
    Stream,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum Location {
    Local,
    Remote,
}

#[derive(Debug, Clone)]
pub enum StreamType {
    Pre,
    Live,
}

impl MediaType {
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
            MediaType::Image(Location::Remote) // TODO: pre-download to local???
        } else if VIDEO_EXTENSIONS
            .iter()
            .any(|&ext| url.ends_with(&format!(".{}", ext)))
        {
            MediaType::Video(Location::Remote)
        } else if STREAM_PROTOCOLS
            .iter()
            .any(|&protocol| url.contains(protocol))
        {
            MediaType::Stream
        } else {
            MediaType::Unknown
        }
    }
}
// ----------------------------

// #[derive(Debug, Clone)]
// pub enum MediaType {
//     Image(ImageType),
//     Video(VideoType),
//     Audio(AudioType),
//     Unknown,
// }

// #[derive(Debug, Clone)]
// pub enum ImageType {
//     Local,
//     Remote,
// }

// #[derive(Debug, Clone)]
// pub enum VideoType {
//     Local,
//     Remote,
//     Stream,
// }

// #[derive(Debug, Clone)]
// pub enum AudioType {
//     Local,
//     Remote,
// }

// impl MediaType {
//     pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Self {
//         let extension = path
//             .as_ref()
//             .extension()
//             .and_then(|ext| ext.to_str())
//             .unwrap_or("")
//             .to_lowercase();

//         if IMAGE_EXTENSIONS.contains(&extension.as_str()) {
//             MediaType::Image(ImageType::Local)
//         } else if VIDEO_EXTENSIONS.contains(&extension.as_str()) {
//             MediaType::Video(VideoType::Local)
//         } else if AUDIO_EXTENSIONS.contains(&extension.as_str()) {
//             MediaType::Audio(AudioType::Local)
//         } else {
//             MediaType::Unknown
//         }
//     }

//     pub fn from_url(url: &str) -> Self {
//         if IMAGE_EXTENSIONS
//             .iter()
//             .any(|&ext| url.ends_with(&format!(".{}", ext)))
//         {
//             MediaType::Image(ImageType::Remote)
//         } else if VIDEO_EXTENSIONS
//             .iter()
//             .any(|&ext| url.ends_with(&format!(".{}", ext)))
//         {
//             MediaType::Video(VideoType::Remote)
//         } else if STREAM_PROTOCOLS
//             .iter()
//             .any(|&protocol| url.contains(protocol))
//         {
//             MediaType::Video(VideoType::Stream)
//         } else {
//             MediaType::Unknown
//         }
//     }
// }
