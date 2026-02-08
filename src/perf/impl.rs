use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

/// Global monotonic counter — gives each *new* key a birth order.
static PERF_SEQ: AtomicU64 = AtomicU64::new(0);

/// Global performance aggregator.
///
/// Every `perf!` call writes directly here via a single mutex lock.
struct GlobalPerf {
    entries: Vec<(String, Vec<Duration>)>,
    index: HashMap<String, (usize, u64)>, // (entry_idx, birth_seq)
}

impl GlobalPerf {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &str, duration: Duration) {
        if let Some(&(idx, _)) = self.index.get(key) {
            self.entries[idx].1.push(duration);
        } else {
            let idx = self.entries.len();
            let seq = PERF_SEQ.fetch_add(1, Ordering::Relaxed);
            self.index.insert(key.to_string(), (idx, seq));
            self.entries.push((key.to_string(), vec![duration]));
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
    }
}

static GLOBAL_PERF: LazyLock<Mutex<GlobalPerf>> = LazyLock::new(|| Mutex::new(GlobalPerf::new()));

/// Record a single measurement (called from the `perf!` macro).
#[inline(always)]
#[doc(hidden)]
pub fn __perf_record(key: &str, duration: Duration) {
    if let Ok(mut g) = GLOBAL_PERF.lock() {
        g.push(key, duration);
    }
}

/// Return a snapshot of all collected entries sorted by first-seen order.
pub(crate) fn perf_collect() -> Vec<(String, Vec<Duration>)> {
    if let Ok(g) = GLOBAL_PERF.lock() {
        let mut entries = g.entries.clone();
        entries.sort_by_key(|(key, _)| g.index.get(key).map_or(u64::MAX, |&(_, seq)| seq));
        entries
    } else {
        Vec::new()
    }
}

/// Clear all performance data.
pub fn perf_clear() {
    if let Ok(mut g) = GLOBAL_PERF.lock() {
        g.clear();
    }
}

/// Unified performance measurement macro.
#[macro_export]
#[doc(hidden)]
macro_rules! perf {
    ($key:expr, $code:expr) => {{
        let __perf_start = std::time::Instant::now();
        let __perf_ret = $code;
        $crate::__perf_record($key, __perf_start.elapsed());
        __perf_ret
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_perf_macro() {
        perf_clear();
        let result = perf!("test::op", {
            std::thread::sleep(Duration::from_millis(1));
            42
        });
        assert_eq!(result, 42);

        let data = perf_collect();
        assert!(data.iter().any(|(k, _)| k == "test::op"));
        perf_clear();
    }

    #[test]
    fn test_thread_safety() {
        perf_clear();
        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    perf!(&format!("thread::task_{i}"), {
                        std::thread::sleep(Duration::from_millis(1));
                    });
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let data = perf_collect();
        assert!(!data.is_empty());
        perf_clear();
    }

    #[test]
    fn test_clear() {
        perf_clear();
        perf!("clear::test", { 1 + 1 });
        perf_clear();
        let data = perf_collect();
        assert!(data.is_empty());
    }
}
