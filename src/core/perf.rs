//! Performance Analysis Module
use std::time::Duration;

use crate::global_ts_manager;

/// High-performance monitoring interface
#[derive(Debug, Clone)]
pub struct Perf;

impl Perf {
    /// Enable performance monitoring
    pub fn enable() {
        log::info!("🚀 usls Performance monitoring enabled");
    }

    /// Disable performance monitoring
    pub fn disable() {
        log::info!("⏸️  usls Performance monitoring disabled");
    }

    /// Check if monitoring is enabled
    pub fn is_enabled() -> bool {
        global_ts_manager().is_enabled()
    }

    /// Get performance statistics
    pub fn stats() -> Option<(Duration, Duration, usize)> {
        global_ts_manager().global_stats()
    }

    /// Clear all performance data
    pub fn clear() {
        global_ts_manager().clear_all();
        log::info!("🧹 Performance data cleared");
    }

    /// Export performance data as JSON
    #[cfg(feature = "serde")]
    pub fn export_json() -> Result<String, Box<dyn std::error::Error>> {
        use serde_json::json;
        let stats = Self::stats();
        let data = json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "enabled": Self::is_enabled(),
            "global_stats": stats.map(|(total, avg, count)| json!({
                "total_time_ms": total.as_millis(),
                "average_time_ms": avg.as_millis(),
                "operation_count": count
            }))
        });
        Ok(serde_json::to_string_pretty(&data)?)
    }

    /// Show traditional table format
    pub fn table() {
        global_ts_manager().print_enhanced_summary();
    }

    /// Show ASCII chart visualization
    pub fn ascii() {
        Self::show_ascii_chart();
    }

    /// Show performance data with optional table
    pub fn show(show_table: bool) {
        Self::show_ascii_chart();
        if show_table {
            Self::table();
        }
    }

    /// Show enhanced ASCII performance chart with detailed breakdown
    fn show_ascii_chart() {
        let manager = global_ts_manager();
        let mut has_data = false;

        // Detailed data collection - using Duration to preserve precision
        let mut dataloader_details = Vec::new();
        let mut model_details = Vec::new();
        let mut annotator_details = Vec::new();

        let mut dataloader_total = Duration::ZERO;
        let mut _model_total = Duration::ZERO;
        let mut annotator_total = Duration::ZERO;
        let mut _total_images = 0;

        // Collect global data with detailed breakdown
        if let Ok(ts) = manager.global().lock() {
            if !ts.is_empty() {
                has_data = true;
                for name in ts.get_names() {
                    let durations = &ts[name.as_str()];
                    let total_duration = durations.iter().sum::<Duration>();
                    let count = durations.len();
                    let avg_duration = if count > 0 {
                        total_duration / count as u32
                    } else {
                        Duration::ZERO
                    };

                    if name.contains("load") || name.contains("read") {
                        dataloader_total += total_duration;
                        dataloader_details.push((
                            name.clone(),
                            total_duration,
                            count,
                            avg_duration,
                        ));
                        if name.contains("image") || name.contains("load") {
                            _total_images += count;
                        }
                    } else if name.contains("annotate")
                        || name.contains("draw")
                        || name.contains("render")
                    {
                        annotator_total += total_duration;
                        annotator_details.push((name.clone(), total_duration, count, avg_duration));
                    } else {
                        _model_total += total_duration;
                        model_details.push((name.clone(), total_duration, count, avg_duration));
                    }
                }
            }
        }

        // Collect module data with detailed breakdown - dynamically get all registered modules
        let module_names = manager.get_module_names();

        for module_name in module_names {
            if let Ok(ts) = manager.module(&module_name).lock() {
                if !ts.is_empty() {
                    has_data = true;
                    for name in ts.get_names() {
                        let durations = &ts[name.as_str()];
                        let total_duration = durations.iter().sum::<Duration>();
                        let count = durations.len();
                        let avg_duration = if count > 0 {
                            total_duration / count as u32
                        } else {
                            Duration::ZERO
                        };
                        let full_name = format!("{}::{}", module_name, name);

                        // Categorize based on module type
                        if module_name == "DATALOADER"
                            || name.contains("load")
                            || name.contains("read")
                        {
                            dataloader_total += total_duration;
                            dataloader_details.push((
                                full_name,
                                total_duration,
                                count,
                                avg_duration,
                            ));
                            if name.contains("image") || name.contains("load") {
                                _total_images += count;
                            }
                        } else if module_name == "ANNOTATOR"
                            || name.contains("annotate")
                            || name.contains("draw")
                            || name.contains("render")
                        {
                            annotator_total += total_duration;
                            annotator_details.push((
                                full_name,
                                total_duration,
                                count,
                                avg_duration,
                            ));
                        } else if module_name != "ENGINE" {
                            // All non-system modules are considered model modules
                            _model_total += total_duration;
                            model_details.push((full_name, total_duration, count, avg_duration));
                        }
                    }
                }
            }
        }

        if !has_data {
            println!("\n📊 No performance data available");
            return;
        }

        println!("\n\n🚀 usls Performance Analysis Chart");
        println!("═══════════════════════════════════════════════════════════════\n");

        // Extract unique model names from model_details
        let mut model_totals: std::collections::HashMap<String, Duration> =
            std::collections::HashMap::new();

        // Preserve Duration precision throughout calculation
        for (name, total_duration, _, _) in &model_details {
            if name.contains("::") {
                // Extract from MODULE::task format
                let parts: Vec<&str> = name.split("::").collect();
                if parts.len() >= 2 {
                    let module_name = parts[0];

                    *model_totals
                        .entry(module_name.to_string())
                        .or_insert(Duration::ZERO) += *total_duration;
                }
            }
        }

        // Build main category data with individual models
        let mut main_data: Vec<(String, Duration)> =
            vec![("🔍 Data Loading".to_string(), dataloader_total)];

        // Add each model as a separate category
        let mut sorted_models: Vec<_> = model_totals.into_iter().collect();
        sorted_models.sort_by(|a, b| b.1.cmp(&a.1));

        for (model_name, total_time) in sorted_models {
            main_data.push((format!("🤖 {}", model_name), total_time));
        }

        main_data.push(("🎨 Visualization".to_string(), annotator_total));

        let main_data: Vec<_> = main_data
            .into_iter()
            .filter(|(_, v)| *v > Duration::ZERO)
            .collect();

        if main_data.is_empty() {
            println!(" No categorized performance data available");
            return;
        }

        let max_value = main_data
            .iter()
            .map(|(_, v)| *v)
            .max()
            .unwrap_or(Duration::ZERO);
        let chart_width = 40;

        // Calculate total time for percentage calculation
        let total_time: Duration = main_data.iter().map(|(_, v)| *v).sum();

        // Calculate the maximum display width for intelligent alignment
        // Use character count instead of byte length for proper emoji handling
        let mut max_display_width = main_data
            .iter()
            .map(|(n, _)| n.chars().count())
            .max()
            .unwrap_or(0);

        // Also consider sub-item lengths for proper alignment
        for (name, _) in &main_data {
            if !name.contains("Data Loading") && !name.contains("Visualization") {
                // Extract model name without emoji for filtering
                let model_name = if let Some(stripped) = name.strip_prefix("🤖 ") {
                    stripped // Skip "🤖 " prefix
                } else {
                    name
                };

                // Calculate sub-item lengths for this model using the same logic as print_tree_breakdown
                let model_filter_lower = model_name.to_lowercase();
                for (detail_name, _, _, _) in &model_details {
                    if let Some(module_end) = detail_name.find("::") {
                        let module_name = &detail_name[..module_end];
                        if module_name.to_lowercase() == model_filter_lower {
                            let task_name = &detail_name[module_end + 2..];
                            let formatted_task = format_task_with_emoji(task_name);
                            // Format: " ├─ 🧬 textual-inference" or " └─ 🧬 textual-inference"
                            let tree_prefix = format!(" ├─ {}", formatted_task);
                            max_display_width = max_display_width.max(tree_prefix.chars().count());
                        }
                    }
                }
            }
        }

        // Dynamic minimum width: longest item + 1 for better spacing
        let alignment_width = max_display_width + 1;

        // Find the longest bar length for alignment
        let max_bar_length = main_data
            .iter()
            .map(|(_, value)| {
                if max_value > Duration::ZERO {
                    let ratio = value.as_nanos() as f64 / max_value.as_nanos() as f64;
                    let calculated_length = (ratio * chart_width as f64) as usize;
                    calculated_length.max(3) // Ensure minimum visibility
                } else {
                    3
                }
            })
            .max()
            .unwrap_or(chart_width);

        for (name, value) in &main_data {
            // Ensure minimum bar length for visibility
            let min_main_bar_length = 3;
            let bar_length = if max_value > Duration::ZERO {
                let ratio = value.as_nanos() as f64 / max_value.as_nanos() as f64;
                let calculated_length = (ratio * chart_width as f64) as usize;
                calculated_length.max(min_main_bar_length) // Ensure minimum visibility
            } else {
                min_main_bar_length
            };
            let bar = "█".repeat(bar_length);

            // Calculate percentage
            let percentage = if total_time > Duration::ZERO {
                (value.as_nanos() as f64 / total_time.as_nanos() as f64) * 100.0
            } else {
                0.0
            };

            // Calculate padding to align values from the longest bar position
            let value_start_position = alignment_width + 1 + max_bar_length; // +1 for the │ separator
            let current_position = alignment_width + 1 + bar_length;
            let base_padding = if value_start_position > current_position {
                value_start_position - current_position
            } else {
                1 // At least one space
            };
            // Add extra space for non-longest bars
            let extra_padding = if bar_length < max_bar_length { 1 } else { 0 };
            let padding = " ".repeat(base_padding + extra_padding);

            println!(
                "{:<width$}│{}{}{:.3?} ({:.2}%)",
                name,
                bar,
                padding,
                value,
                percentage,
                width = alignment_width,
            );

            // Add tree-style breakdown for model inference categories
            if !name.contains("Data Loading") && !name.contains("Visualization") {
                // Extract model name without emoji for filtering
                let model_name = if let Some(stripped) = name.strip_prefix("🤖 ") {
                    stripped // Skip "🤖 " prefix
                } else {
                    name
                };
                print_tree_breakdown(
                    &model_details,
                    model_name,
                    max_value,
                    chart_width,
                    alignment_width,
                );
            }
        }

        // Use the already calculated max_bar_length for scale line
        println!(
            "{:<width$}└{}",
            "",
            "─".repeat(max_bar_length),
            width = alignment_width + 1
        );
        println!(
            "{:<width$}0{:>bar_width$}  {:.3?}",
            "",
            "",
            total_time,
            width = alignment_width + 1,
            bar_width = max_bar_length.saturating_sub(3), // -3 to account for the extra 2 spaces
        );

        println!();
        println!();
    }
}

// Helper function to format task name with appropriate emoji
fn format_task_with_emoji(task_name: &str) -> String {
    use rand::prelude::*;
    let lower = task_name.to_lowercase();
    let fallback_emojis = ["🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "⚫️", "⚪️", "🟤"];
    let emoji = match lower.as_str() {
        name if name.contains("preprocess") => "🔧",
        name if name.contains("postprocess") => "📦",
        name if name.contains("inference") || name.contains("forward") => "🧬",
        name if name.contains("generate") => "🎲",
        _ => fallback_emojis.choose(&mut rand::rng()).map_or("🟢", |v| v),
    };
    format!("{} {}", emoji, task_name)
}

// Helper function to print tree-style model breakdown
fn print_tree_breakdown(
    model_details: &[(String, Duration, usize, Duration)],
    model_filter: &str,
    max_value: Duration,
    chart_width: usize,
    alignment_width: usize,
) {
    // Collect tasks that belong to this model
    let mut tasks: Vec<(String, Duration)> = Vec::new();
    let model_filter_lower = model_filter.to_lowercase();

    for (name, _total_duration, _count, avg_duration) in model_details {
        // Check if this task belongs to the current model
        // Format: "module::task" where module matches model_filter (case insensitive)
        if let Some(module_end) = name.find("::") {
            let module_name = &name[..module_end];
            if module_name.to_lowercase() == model_filter_lower {
                // Convert to display format: MODULE::task with appropriate emoji
                let task_name = &name[module_end + 2..];
                let formatted_task = format_task_with_emoji(task_name);
                tasks.push((formatted_task, *avg_duration));
            }
        }
    }

    if tasks.is_empty() {
        return;
    }

    // Sort by time (descending)
    tasks.sort_by(|a, b| b.1.cmp(&a.1));

    // Print tree-style breakdown using the same scale as main chart
    for (i, (task_name, avg_time)) in tasks.iter().enumerate() {
        let is_last = i == tasks.len() - 1;
        let branch = if is_last { "└─" } else { "├─" };

        // Calculate bar length using the same scale as main chart
        let min_bar_length = 8; // Minimum bar length for visibility
        let bar_length = if max_value > Duration::ZERO {
            let ratio = avg_time.as_nanos() as f64 / max_value.as_nanos() as f64;
            let calculated_length = (ratio * chart_width as f64) as usize;
            calculated_length.max(min_bar_length) // Ensure minimum visibility
        } else {
            min_bar_length
        };
        let bar = "█".repeat(bar_length);

        // Calculate consistent spacing for tree items with intelligent alignment
        let tree_prefix = format!(" {} {}", branch, task_name);

        // Calculate padding to align values from the longest bar position
        let value_start_position = alignment_width + 1 + chart_width; // +1 for the │ separator
        let current_position = alignment_width + 1 + bar_length;
        let base_padding = if value_start_position > current_position {
            value_start_position - current_position
        } else {
            1 // At least one space
        };
        // Add extra space for non-longest bars (sub-items always get extra space)
        let extra_padding = 1;
        let padding = " ".repeat(base_padding + extra_padding);

        println!(
            "{:<width$}│{}{}{:.3?}",
            tree_prefix,
            bar,
            padding,
            avg_time,
            width = alignment_width
        );
    }
}

/// Show performance data with optional table
/// Default shows ASCII chart, set show_table=true to include detailed table
pub fn perf(show_table: bool) {
    Perf::show(show_table);
}

/// Show traditional table format
pub fn table() {
    Perf::table();
}

/// Show ASCII chart visualization
pub fn ascii() {
    Perf::ascii();
}

/// Enable performance monitoring
pub fn enable() {
    Perf::enable();
}

/// Disable performance monitoring
pub fn disable() {
    Perf::disable();
}

/// Check if monitoring is enabled
pub fn is_enabled() -> bool {
    Perf::is_enabled()
}

/// Get performance statistics
pub fn stats() -> Option<(Duration, Duration, usize)> {
    Perf::stats()
}

/// Clear all performance data
pub fn clear() {
    Perf::clear();
}

/// Export performance data to JSON
#[cfg(feature = "serde")]
pub fn export_json() -> Result<String, Box<dyn std::error::Error>> {
    Perf::export_json()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_perf_interface() {
        assert!(is_enabled());

        // Simulate operation
        crate::elapsed_global!("test_op", {
            thread::sleep(Duration::from_millis(1));
        });

        // Test all visualization modes
        perf(false); // ASCII only
        perf(true); // ASCII + table
        ascii();
        table();

        clear();
    }
}
