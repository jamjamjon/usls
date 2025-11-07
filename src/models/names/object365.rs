use std::sync::LazyLock;

/// Object365 dataset labels without the leading `background` class (365 categories).
pub static NAMES_OBJECT365: LazyLock<&'static [&'static str]> = LazyLock::new(|| {
    let vec: Vec<&'static str> = include_str!("object365.txt")
        .lines()
        .filter(|s| !s.is_empty())
        .collect();
    Box::leak(vec.into_boxed_slice())
});

/// Object365 dataset labels including `background` (366 categories).
/// Built by prepending `background` to `NAMES_OBJECT365` for compatibility.
pub static NAMES_OBJECT365_366: LazyLock<&'static [&'static str]> = LazyLock::new(|| {
    let mut vec = Vec::with_capacity(NAMES_OBJECT365.len() + 1);
    vec.push("background");
    vec.extend(*NAMES_OBJECT365);
    Box::leak(vec.into_boxed_slice())
});
