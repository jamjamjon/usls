use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::core::ts::Ts;

/// Global singleton Ts manager for performance monitoring
/// Provides thread-safe access to global and module-level performance data
pub struct GlobalTsManager {
    /// Global performance data shared across all modules
    global_ts: Arc<Mutex<Ts>>,
    /// Module-specific performance data
    module_ts: Arc<Mutex<HashMap<String, Ts>>>,
    /// Performance monitoring enabled flag (for zero-overhead when disabled)
    enabled: bool,
}

impl GlobalTsManager {
    /// Create a new GlobalTsManager instance
    fn new() -> Self {
        Self {
            global_ts: Arc::new(Mutex::new(Ts::default())),
            module_ts: Arc::new(Mutex::new(HashMap::new())),
            enabled: true, // Can be controlled via env var or config
        }
    }

    /// Check if performance monitoring is enabled
    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable performance monitoring
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get the global Ts instance
    pub fn global(&self) -> Arc<Mutex<Ts>> {
        Arc::clone(&self.global_ts)
    }

    /// Get or create a module-specific Ts instance
    pub fn module(&self, module_name: &str) -> Arc<Mutex<Ts>> {
        let mut modules = self.module_ts.lock().unwrap();
        if !modules.contains_key(module_name) {
            modules.insert(module_name.to_string(), Ts::default());
        }
        // Return a cloned Arc to the specific module's Ts
        // Note: This is a simplified approach. In a real implementation,
        // we might want to return a wrapper that provides direct access
        Arc::new(Mutex::new(modules.get(module_name).unwrap().clone()))
    }

    /// Get all registered module names
    pub fn get_module_names(&self) -> Vec<String> {
        if let Ok(modules) = self.module_ts.lock() {
            modules.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Push timing data to global Ts (with early return if disabled)
    #[inline(always)]
    pub fn push_global(&self, label: &str, duration: Duration) {
        if !self.enabled {
            return;
        }
        if let Ok(mut ts) = self.global_ts.lock() {
            ts.push(label, duration);
        }
    }

    /// Push timing data to module-specific Ts (with early return if disabled)
    #[inline(always)]
    pub fn push_module(&self, module_name: &str, label: &str, duration: Duration) {
        if !self.enabled {
            return;
        }
        // Push timing data to module
        if let Ok(mut modules) = self.module_ts.lock() {
            let ts = modules
                .entry(module_name.to_string())
                .or_insert_with(Ts::default);
            ts.push(label, duration);
            // Successfully pushed timing data
        }
    }

    /// Print global performance summary
    pub fn print_global_summary(&self) {
        if let Ok(ts) = self.global_ts.lock() {
            println!("\n=== Global Performance Summary ===");
            ts.summary();
        }
    }

    /// Print module-specific performance summary
    pub fn print_module_summary(&self, module_name: &str) {
        if let Ok(modules) = self.module_ts.lock() {
            // Available modules for summary
            if let Some(ts) = modules.get(module_name) {
                println!("\n=== {} Module Performance Summary ===", module_name);
                // Module summary
                ts.summary();
            } else {
                // Module not found
            }
        }
    }

    /// Print all performance summaries in a unified format with enhanced visualization
    pub fn print_all_summaries(&self) {
        self.print_enhanced_summary();
    }

    /// Enhanced performance summary with better visualization and categorization
    pub fn print_enhanced_summary(&self) {
        // Collect all timing data with categorization
        let pipeline_data = Vec::new();
        let mut dataloader_data = Vec::new();
        let mut model_data = Vec::new();
        let mut annotator_data = Vec::new();

        // Add global data
        if let Ok(ts) = self.global_ts.lock() {
            if !ts.is_empty() {
                for name in ts.get_names() {
                    let durations = &ts[name.as_str()];
                    let count = durations.len();
                    let avg = ts.avg_by_key(name).unwrap_or_default();
                    let min = durations.iter().min().copied().unwrap_or_default();
                    let max = durations.iter().max().copied().unwrap_or_default();
                    let total = durations.iter().sum::<Duration>();

                    let entry = (name.to_string(), count, avg, min, max, total);

                    // Categorize based on task name
                    if name.contains("dataloader") || name.contains("load") || name.contains("read")
                    {
                        dataloader_data.push(entry);
                    } else if name.contains("annotate")
                        || name.contains("draw")
                        || name.contains("render")
                    {
                        annotator_data.push(entry);
                    } else {
                        // All other operations are considered model inference related
                        model_data.push(entry);
                    }
                }
            }
        }

        // Add module data
        if let Ok(modules) = self.module_ts.lock() {
            for (module_name, ts) in modules.iter() {
                if !ts.is_empty() {
                    for name in ts.get_names() {
                        let durations = &ts[name.as_str()];
                        let count = durations.len();
                        let avg = ts.avg_by_key(name).unwrap_or_default();
                        let min = durations.iter().min().copied().unwrap_or_default();
                        let max = durations.iter().max().copied().unwrap_or_default();
                        let total = durations.iter().sum::<Duration>();

                        let entry = (
                            format!("{}::{}", module_name.to_uppercase(), name),
                            count,
                            avg,
                            min,
                            max,
                            total,
                        );

                        // Categorize based on module and task name
                        if module_name.to_lowercase().contains("dataloader")
                            || name.contains("load")
                            || name.contains("read")
                        {
                            dataloader_data.push(entry);
                        } else if module_name.to_lowercase().contains("annotator")
                            || name.contains("annotate")
                            || name.contains("draw")
                        {
                            annotator_data.push(entry);
                        } else {
                            // All other operations are considered model inference related
                            model_data.push(entry);
                        }
                    }
                }
            }
        }

        // Combine all data for total calculation
        let all_data: Vec<_> = pipeline_data
            .iter()
            .chain(dataloader_data.iter())
            .chain(model_data.iter())
            .chain(annotator_data.iter())
            .collect();

        if all_data.is_empty() {
            println!("📊 No performance data available.");
            return;
        }

        println!("\n🚀 usls Performance Analysis Dashboard");
        println!("═══════════════════════════════════════════════════════════════");

        // Print categorized sections
        self.print_category("📁 Data Loading", &dataloader_data);
        self.print_category("🧠 Model Inference", &model_data);
        self.print_category("🎨 Visualization", &annotator_data);
    }

    /// Print a categorized section of performance data
    fn print_category(
        &self,
        title: &str,
        data: &[(String, usize, Duration, Duration, Duration, Duration)],
    ) {
        if data.is_empty() {
            return;
        }

        println!("\n{}", title);
        println!("───────────────────────────────────────────────────────────────");

        // Calculate column widths
        let decimal_places = 3;
        let width_task = data
            .iter()
            .map(|(task, _, _, _, _, _)| task.len())
            .max()
            .unwrap_or(20)
            .max(20)
            + 2;
        let width_count = 8;
        let width_time = 12;

        // Print header
        println!(
            " {:<width_task$} {:<width_count$} {:<width_time$} {:<width_time$} {:<width_time$} {:<width_time$}",
            "Task", "Count", "Mean", "Min", "Max", "Total"
        );
        println!(
            " {}",
            "-".repeat(width_task + width_count + width_time * 4 + 8)
        );

        // Print data rows
        for (task, count, avg, min, max, total) in data {
            println!(
                " {:<width_task$} {:<width_count$} {:<width_time$.decimal_places$?} {:<width_time$.decimal_places$?} {:<width_time$.decimal_places$?} {:<width_time$.decimal_places$?}",
                task, count, avg, min, max, total
            );
        }

        // Print category total
        let category_total: Duration = data.iter().map(|(_, _, _, _, _, total)| *total).sum();
        let category_count: usize = data.iter().map(|(_, count, _, _, _, _)| *count).sum();
        println!(
            " {}",
            "-".repeat(width_task + width_count + width_time * 4 + 8)
        );
        println!(
            " {:<width_task$} {:<width_count$} {:<width_time$} {:<width_time$} {:<width_time$} {:<width_time$.decimal_places$?}",
            "(Category Total)", category_count, "-", "-", "-", category_total
        );
    }

    /// Get global statistics
    pub fn global_stats(&self) -> Option<(Duration, Duration, usize)> {
        if let Ok(ts) = self.global_ts.lock() {
            if !ts.is_empty() {
                let total = ts.sum();
                let avg = ts.avg().ok()?;
                let count = ts.names().len();
                return Some((total, avg, count));
            }
        }
        None
    }

    /// Clear all performance data
    pub fn clear_all(&self) {
        if let Ok(mut ts) = self.global_ts.lock() {
            ts.clear();
        }
        if let Ok(mut modules) = self.module_ts.lock() {
            modules.clear();
        }
    }
}

/// Global singleton instance
static GLOBAL_TS_MANAGER: Lazy<GlobalTsManager> = Lazy::new(GlobalTsManager::new);

/// Get the global Ts manager instance
pub fn global_ts_manager() -> &'static GlobalTsManager {
    &GLOBAL_TS_MANAGER
}

/// Convenient access to global Ts
pub fn global_ts() -> Arc<Mutex<Ts>> {
    global_ts_manager().global()
}

/// Convenient access to module Ts
pub fn module_ts(module_name: &str) -> Arc<Mutex<Ts>> {
    global_ts_manager().module(module_name)
}

/// High-performance global timing macro with zero overhead when disabled
#[macro_export]
macro_rules! elapsed_global {
    ($label:expr, $code:expr) => {{
        let manager = $crate::global_ts_manager();
        if manager.is_enabled() {
            let t = std::time::Instant::now();
            let ret = $code;
            let duration = t.elapsed();
            manager.push_global($label, duration);
            ret
        } else {
            $code
        }
    }};
}

/// High-performance module timing macro with zero overhead when disabled
#[macro_export]
macro_rules! elapsed_module {
    ($module:expr, $label:expr, $code:expr) => {{
        let manager = $crate::global_ts_manager();
        if manager.is_enabled() {
            let t = std::time::Instant::now();
            let ret = $code;
            let duration = t.elapsed();
            manager.push_module($module, $label, duration);
            ret
        } else {
            $code
        }
    }};
}

/// Dataloader-specific timing macro
#[macro_export]
macro_rules! elapsed_dataloader {
    ($label:expr, $code:expr) => {{
        $crate::elapsed_module!("DATALOADER", $label, $code)
    }};
}

/// Annotator-specific timing macro
#[macro_export]
macro_rules! elapsed_annotator {
    ($label:expr, $code:expr) => {{
        $crate::elapsed_module!("ANNOTATOR", $label, $code)
    }};
}

/// Engine-specific timing macro
#[macro_export]
macro_rules! elapsed_engine {
    ($label:expr, $code:expr) => {{
        $crate::elapsed_module!("ENGINE", $label, $code)
    }};
}

/// Scoped performance timer for RAII-style timing
pub struct ScopedTimer {
    label: String,
    module: Option<String>,
    start: Instant,
}

impl ScopedTimer {
    /// Create a new scoped timer for global timing
    pub fn global(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            module: None,
            start: Instant::now(),
        }
    }

    /// Create a new scoped timer for module timing
    pub fn module(module: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            module: Some(module.into()),
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let manager = global_ts_manager();

        if let Some(ref module) = self.module {
            manager.push_module(module, &self.label, duration);
        } else {
            manager.push_global(&self.label, duration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_global_ts_manager() {
        let manager = global_ts_manager();

        // Test global timing
        manager.push_global("test_task", Duration::from_millis(100));
        manager.push_global("test_task", Duration::from_millis(200));

        // Test module timing
        manager.push_module("yolo", "inference", Duration::from_millis(50));
        manager.push_module("yolo", "postprocess", Duration::from_millis(30));

        // Verify data
        if let Ok(global_ts) = manager.global().lock() {
            assert_eq!(global_ts["test_task"].len(), 2);
        }

        // Test stats
        let stats = manager.global_stats();
        assert!(stats.is_some());
    }

    #[test]
    fn test_macros() {
        // Test global macro
        let result = elapsed_global!("macro_test", {
            std::thread::sleep(Duration::from_millis(1));
            42
        });
        assert_eq!(result, 42);

        // Test module macro
        let result = elapsed_module!("TEST_MODULE", "macro_test", {
            std::thread::sleep(Duration::from_millis(1));
            "hello"
        });
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_thread_safety() {
        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let manager = global_ts_manager();
                    manager.push_global(&format!("task_{}", i), Duration::from_millis(i * 10));
                    manager.push_module(
                        "concurrent_module",
                        &format!("subtask_{}", i),
                        Duration::from_millis(i * 5),
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all data was recorded
        let manager = global_ts_manager();
        if let Ok(global_ts) = manager.global().lock() {
            // Should have at least some tasks recorded
            assert!(!global_ts.is_empty());
        }
    }

    #[test]
    fn test_scoped_timer() {
        {
            let _timer = ScopedTimer::global("scoped_test");
            std::thread::sleep(Duration::from_millis(1));
        } // Timer drops here and records timing

        {
            let _timer = ScopedTimer::module("TEST_MODULE", "scoped_module_test");
            std::thread::sleep(Duration::from_millis(1));
        } // Timer drops here and records timing

        // Verify data was recorded
        let manager = global_ts_manager();
        if let Ok(global_ts) = manager.global().lock() {
            assert!(!global_ts.is_empty());
        }
    }
}
