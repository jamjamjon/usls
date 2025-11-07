//! DOTA aerial imagery categories.

/// DOTA v1.5 (16 classes, includes `container crane`).
pub const NAMES_DOTA_V1_5_16: [&str; 16] = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane",
];

/// DOTA v1.0 (15 classes, excludes `container crane`).
pub const NAMES_DOTA_V1_15: [&str; 15] = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
];
