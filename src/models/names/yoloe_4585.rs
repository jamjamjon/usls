use std::sync::LazyLock;

/// YOLOE model class names (4585 categories).
///
/// This array is loaded from a text file at runtime to reduce compile-time overhead.
/// The labels are used for instance segmentation tasks with the YOLOE model.
///
/// Usage: `&NAMES_YOLOE_4585` (same as other NAMES_* constants)
pub static NAMES_YOLOE_4585: LazyLock<&'static [&'static str]> = LazyLock::new(|| {
    let vec: Vec<&'static str> = include_str!("yoloe_4585.txt")
        .lines()
        .filter(|s| !s.is_empty())
        .collect();
    Box::leak(vec.into_boxed_slice())
});
