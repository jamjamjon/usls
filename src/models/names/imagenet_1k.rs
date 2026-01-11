use std::sync::LazyLock;

/// ImageNet-1K classification labels (1000 categories).
/// Lazily loaded from an embedded text file to keep compile time low.
pub static NAMES_IMAGENET_1K: LazyLock<&'static [&'static str]> = LazyLock::new(|| {
    let vec: Vec<&'static str> = include_str!("imagenet_1k.txt")
        .lines()
        .filter(|s| !s.is_empty())
        .collect();
    Box::leak(vec.into_boxed_slice())
});
