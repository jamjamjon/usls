/// The name of the current crate.
pub(crate) const CRATE_NAME: &str = env!("CARGO_PKG_NAME");
/// Standard prefix length for progress bar formatting.
pub(crate) const PREFIX_LENGTH: usize = 12;
/// Progress bar style for completion with iteration count.
pub(crate) const PROGRESS_BAR_STYLE_FINISH: &str =
    "{prefix:>12.green.bold} {msg} for {human_len} iterations in {elapsed}";
/// Progress bar style for completion with multiplier format.
pub(crate) const PROGRESS_BAR_STYLE_FINISH_2: &str =
    "{prefix:>12.green.bold} {msg} x{human_len} in {elapsed}";
/// Progress bar style for completion with byte size information.
pub(crate) const PROGRESS_BAR_STYLE_FINISH_3: &str =
    "{prefix:>12.green.bold} {msg} ({binary_total_bytes}) in {elapsed}";
/// Progress bar style for ongoing operations with position indicator.
pub(crate) const PROGRESS_BAR_STYLE_CYAN_2: &str =
    "{prefix:>12.cyan.bold} {human_pos}/{human_len} |{bar}| {msg}";
