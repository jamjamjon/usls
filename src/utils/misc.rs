#![allow(dead_code)]

use glob::{glob_with, MatchOptions};
use std::path::{Path, PathBuf};

pub(crate) fn load_paths(
    source: &str,
    exts: Option<&[&str]>,
    sort: bool,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();

    if source.contains('*')
        || source.contains('?')
        || (source.contains('[') && source.contains(']'))
    {
        // Branch B: Glob pattern
        let config = MatchOptions {
            case_sensitive: true,
            require_literal_separator: false,
            require_literal_leading_dot: false,
        };
        paths = glob_with(source, config)?
            .filter_map(|entry| entry.ok())
            .filter(|p| p.is_file())
            .collect();

        // Apply extension filtering if provided
        if let Some(exts) = exts {
            let exts_set: std::collections::HashSet<String> =
                exts.iter().map(|s| s.to_lowercase()).collect();
            paths.retain(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| exts_set.contains(&s.to_lowercase()))
                    .unwrap_or(false)
            });
        }
    } else {
        // Branch A: Direct directory scan (High performance)
        let source_path = Path::new(source);
        if source_path.is_dir() {
            let exts_set: Option<std::collections::HashSet<String>> =
                exts.map(|e| e.iter().map(|s| s.to_lowercase()).collect());

            paths = std::fs::read_dir(source_path)?
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    if path.is_file() {
                        if let Some(ref set) = exts_set {
                            let ext = path.extension()?.to_str()?.to_lowercase();
                            if set.contains(&ext) {
                                return Some(path);
                            }
                        } else {
                            return Some(path);
                        }
                    }
                    None
                })
                .collect();
        } else if source_path.is_file() {
            paths.push(source_path.to_path_buf());
        }
    }

    // Sort results using natural comparison
    if sort {
        paths.sort_by(|a, b| {
            let a = a.file_name().and_then(|s| s.to_str());
            let b = b.file_name().and_then(|s| s.to_str());
            match (a, b) {
                (Some(a), Some(b)) => crate::natural_compare(a, b),
                _ => std::cmp::Ordering::Equal,
            }
        });
    }

    Ok(paths)
}

pub(crate) fn try_fetch_file_stem<P: AsRef<std::path::Path>>(p: P) -> anyhow::Result<String> {
    let p = p.as_ref();
    let stem = p
        .file_stem()
        .ok_or(anyhow::anyhow!(
            "Failed to get the `file_stem` of `model_file`: {p:?}"
        ))?
        .to_str()
        .ok_or(anyhow::anyhow!("Failed to convert from `&OsStr` to `&str`"))?;

    Ok(stem.to_string())
}

/// Formats a byte size into a human-readable string using decimal (base-1000) units.
///
/// # Arguments
/// * `size` - The size in bytes to format
/// * `decimal_places` - Number of decimal places to show in the formatted output
///
/// # Returns
/// A string representing the size with appropriate decimal unit (B, KB, MB, etc.)
///
/// # Example
/// ```ignore
/// let size = 1500000.0;
/// let formatted = human_bytes_decimal(size, 2);
/// assert_eq!(formatted, "1.50 MB");
/// ```
pub(crate) fn human_bytes_decimal(size: f64, decimal_places: usize) -> String {
    const DECIMAL_UNITS: [&str; 7] = ["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    format_bytes_internal(size, 1000.0, &DECIMAL_UNITS, decimal_places)
}

/// Formats a byte size into a human-readable string using binary (base-1024) units.
///
/// # Arguments
/// * `size` - The size in bytes to format
/// * `decimal_places` - Number of decimal places to show in the formatted output
///
/// # Returns
/// A string representing the size with appropriate binary unit (B, KiB, MiB, etc.)
///
/// # Example
/// ```ignore
/// let size = 1024.0 * 1024.0; // 1 MiB in bytes
/// let formatted = human_bytes_binary(size, 2);
/// assert_eq!(formatted, "1.00 MiB");
/// ```
pub(crate) fn human_bytes_binary(size: f64, decimal_places: usize) -> String {
    const BINARY_UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    format_bytes_internal(size, 1024.0, &BINARY_UNITS, decimal_places)
}

fn format_bytes_internal(
    mut size: f64,
    base: f64,
    units: &[&str],
    decimal_places: usize,
) -> String {
    let mut unit_index = 0;
    while size >= base && unit_index < units.len() - 1 {
        size /= base;
        unit_index += 1;
    }

    format!(
        "{:.precision$} {}",
        size,
        units[unit_index],
        precision = decimal_places
    )
}

/// Generates a random alphanumeric string of the specified length.
///
/// # Arguments
/// * `length` - The desired length of the random string
///
/// # Returns
/// A string containing random alphanumeric characters (A-Z, a-z, 0-9).
/// Returns an empty string if length is 0.
///
/// # Example
/// ```ignore
/// let random = generate_random_string(8);
/// assert_eq!(random.len(), 8);
/// // Each character in the string will be alphanumeric
/// assert!(random.chars().all(|c| c.is_ascii_alphanumeric()));
/// ```
pub(crate) fn generate_random_string(length: usize) -> String {
    use rand::{distr::Alphanumeric, rng, Rng};
    if length == 0 {
        return String::new();
    }
    let rng = rng();
    let mut result = String::with_capacity(length);
    result.extend(rng.sample_iter(&Alphanumeric).take(length).map(char::from));
    result
}

/// Generates a timestamp string in the format "YYYYMMDDHHmmSSffffff" with an optional delimiter.
///
/// # Arguments
/// * `delimiter` - Optional string to insert between each component of the timestamp.
///   If None or empty string, components will be joined without delimiter.
///
/// # Returns
/// A string containing the current timestamp with the specified format.
///
/// # Example
/// ```ignore
/// use chrono::TimeZone;
///
/// // Without delimiter
/// let ts = timestamp(None);
/// assert_eq!(ts.len(), 20); // YYYYMMDDHHmmSSffffff
///
/// // With delimiter
/// let ts = timestamp(Some("-"));
/// // Format: YYYY-MM-DD-HH-mm-SS-ffffff
/// assert_eq!(ts.split('-').count(), 7);
/// ```
pub fn timestamp(delimiter: Option<&str>) -> String {
    let delimiter = delimiter.unwrap_or("");
    let format =
        format!("%Y{delimiter}%m{delimiter}%d{delimiter}%H{delimiter}%M{delimiter}%S{delimiter}%f");
    chrono::Local::now().format(&format).to_string()
}

/// Natural sort comparison for strings (handles embedded numbers correctly)
///
/// Compares strings by breaking them into chunks of digits and non-digits,
/// comparing digit chunks numerically and text chunks lexicographically.
///
/// # Examples
/// ```ignore
/// use usls::natural_compare;
///
/// assert!(natural_compare("file1.txt", "file2.txt") == std::cmp::Ordering::Less);
/// assert!(natural_compare("file2.txt", "file10.txt") == std::cmp::Ordering::Less);
/// assert!(natural_compare("img001.jpg", "img100.jpg") == std::cmp::Ordering::Less);
/// ```
pub(crate) fn natural_compare(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let mut a_chars = a.chars().peekable();
    let mut b_chars = b.chars().peekable();

    loop {
        match (a_chars.peek(), b_chars.peek()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(&ca), Some(&cb)) => {
                // Both are digits - compare numerically
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    // Extract full number from both strings
                    let mut num_a = String::new();
                    let mut num_b = String::new();

                    while let Some(&c) = a_chars.peek() {
                        if c.is_ascii_digit() {
                            num_a.push(c);
                            a_chars.next();
                        } else {
                            break;
                        }
                    }

                    while let Some(&c) = b_chars.peek() {
                        if c.is_ascii_digit() {
                            num_b.push(c);
                            b_chars.next();
                        } else {
                            break;
                        }
                    }

                    // Compare as numbers
                    // Handle potential parsing errors by falling back to string comparison
                    match (num_a.parse::<u64>(), num_b.parse::<u64>()) {
                        (Ok(na), Ok(nb)) => match na.cmp(&nb) {
                            Ordering::Equal => continue,
                            other => return other,
                        },
                        // If numbers are too large for u64, compare string length first
                        // then lexicographically (maintains correct ordering for large numbers)
                        _ => match num_a.len().cmp(&num_b.len()) {
                            Ordering::Equal => match num_a.cmp(&num_b) {
                                Ordering::Equal => continue,
                                other => return other,
                            },
                            other => return other,
                        },
                    }
                } else {
                    // At least one is not a digit - compare lexicographically
                    a_chars.next();
                    b_chars.next();
                    match ca.cmp(&cb) {
                        Ordering::Equal => continue,
                        other => return other,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_natural_compare_basic() {
        assert_eq!(natural_compare("file1.txt", "file2.txt"), Ordering::Less);
        assert_eq!(natural_compare("file2.txt", "file10.txt"), Ordering::Less);
        assert_eq!(
            natural_compare("file10.txt", "file2.txt"),
            Ordering::Greater
        );
        assert_eq!(natural_compare("file5.txt", "file5.txt"), Ordering::Equal);
    }

    #[test]
    fn test_natural_compare_leading_zeros() {
        assert_eq!(natural_compare("img001.jpg", "img002.jpg"), Ordering::Less);
        assert_eq!(natural_compare("img001.jpg", "img100.jpg"), Ordering::Less);
        assert_eq!(natural_compare("img099.jpg", "img100.jpg"), Ordering::Less);

        // Leading zeros are preserved in text but compared numerically
        assert_eq!(natural_compare("img01.jpg", "img1.jpg"), Ordering::Equal); // 01 == 1
    }

    #[test]
    fn test_natural_compare_version_numbers() {
        assert_eq!(natural_compare("v1.0.1", "v1.0.2"), Ordering::Less);
        assert_eq!(natural_compare("v1.0.9", "v1.0.10"), Ordering::Less);
        assert_eq!(natural_compare("v1.9.0", "v1.10.0"), Ordering::Less);
        assert_eq!(natural_compare("v2.0.0", "v1.99.99"), Ordering::Greater);
    }

    #[test]
    fn test_natural_compare_mixed_content() {
        assert_eq!(natural_compare("test", "test123"), Ordering::Less);
        assert_eq!(natural_compare("test123", "test"), Ordering::Greater);
        assert_eq!(natural_compare("abc123def", "abc123def"), Ordering::Equal);
        assert_eq!(natural_compare("abc123def", "abc124def"), Ordering::Less);
    }

    #[test]
    fn test_natural_compare_edge_cases() {
        // Empty strings
        assert_eq!(natural_compare("", ""), Ordering::Equal);
        assert_eq!(natural_compare("", "a"), Ordering::Less);
        assert_eq!(natural_compare("a", ""), Ordering::Greater);

        // Pure numbers
        assert_eq!(natural_compare("1", "2"), Ordering::Less);
        assert_eq!(natural_compare("9", "10"), Ordering::Less);
        assert_eq!(natural_compare("100", "99"), Ordering::Greater);

        // Pure text
        assert_eq!(natural_compare("apple", "banana"), Ordering::Less);
        assert_eq!(natural_compare("zebra", "apple"), Ordering::Greater);
    }

    #[test]
    fn test_natural_compare_complex_filenames() {
        // Common image sequence patterns
        assert_eq!(
            natural_compare("frame_0001.png", "frame_0010.png"),
            Ordering::Less
        );
        assert_eq!(
            natural_compare("shot1_take5.mp4", "shot1_take15.mp4"),
            Ordering::Less
        );
        assert_eq!(
            natural_compare("scene2_v3.mov", "scene10_v1.mov"),
            Ordering::Less
        );

        // Dataset naming
        assert_eq!(
            natural_compare("data_batch_1", "data_batch_10"),
            Ordering::Less
        );
        assert_eq!(
            natural_compare("train_001.jpg", "train_1000.jpg"),
            Ordering::Less
        );
    }

    #[test]
    fn test_natural_compare_large_numbers() {
        // Numbers larger than u64::MAX
        let large_a = "file99999999999999999999.txt";
        let large_b = "file100000000000000000000.txt";
        assert_eq!(natural_compare(large_a, large_b), Ordering::Less);

        // Same length, different values
        assert_eq!(
            natural_compare(
                "file12345678901234567890.txt",
                "file12345678901234567891.txt"
            ),
            Ordering::Less
        );
    }

    #[test]
    fn test_natural_compare_multiple_numbers() {
        // Multiple number sequences in one string
        assert_eq!(natural_compare("v1.2.3", "v1.2.10"), Ordering::Less);
        assert_eq!(
            natural_compare("chapter1_page5", "chapter1_page15"),
            Ordering::Less
        );
        assert_eq!(natural_compare("2024_01_05", "2024_01_15"), Ordering::Less);
        assert_eq!(natural_compare("2024_12_31", "2025_01_01"), Ordering::Less);
    }

    #[test]
    fn test_natural_compare_case_sensitivity() {
        // Case-sensitive comparison (standard behavior)
        assert_eq!(natural_compare("File1.txt", "file1.txt"), Ordering::Less); // 'F' < 'f'
        assert_eq!(natural_compare("ABC", "abc"), Ordering::Less);
    }

    #[test]
    fn test_natural_sorting() {
        let mut files = vec![
            "file10.txt",
            "file2.txt",
            "file1.txt",
            "file20.txt",
            "file3.txt",
        ];

        files.sort_by(|a, b| natural_compare(a, b));

        assert_eq!(
            files,
            vec![
                "file1.txt",
                "file2.txt",
                "file3.txt",
                "file10.txt",
                "file20.txt"
            ]
        );
    }
}
