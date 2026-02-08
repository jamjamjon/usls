//! Performance chart & dashboard rendering.

use std::time::Duration;

use super::r#impl::perf_collect;

/// Approximate terminal display width of a string.
/// ASCII = 1, zero-width selectors = 0, box-drawing/block = 1, emoji/CJK = 2.
fn display_width(s: &str) -> usize {
    s.chars()
        .map(|c| {
            if c.is_ascii() {
                return 1;
            }
            let cp = c as u32;
            match cp {
                // Zero-width: variation selectors, ZWJ, ZWNBSP
                0xFE00..=0xFE0F | 0x200B..=0x200D | 0x20E3 | 0xE0100..=0xE01EF => 0,
                // Halfwidth: Latin-1 Supplement .. Latin Extended-B (µ, accented letters)
                0x0080..=0x024F => 1,
                // Halfwidth: General Punctuation, Symbols, Arrows, Math, Technical,
                //            Box Drawing (├└─│═), Block Elements (█)
                0x2000..=0x259F => 1,
                // Everything else: emoji, CJK, Misc Symbols (⚫⭐), etc.
                _ => 2,
            }
        })
        .sum()
}

/// Pad `s` with trailing spaces so its display width equals `target`.
fn pad_to(s: &str, target: usize) -> String {
    let w = display_width(s);
    if w >= target {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(target - w))
    }
}

/// Normalize an emoji string to exactly `target_dw` display columns by padding with spaces.
/// This ensures ALL emojis occupy the same fixed width regardless of variation selectors,
/// ZWJ sequences, etc.
fn fixed_emoji(emoji: &str, target_dw: usize) -> String {
    let w = display_width(emoji);
    if w >= target_dw {
        emoji.to_string()
    } else {
        format!("{emoji}{}", " ".repeat(target_dw - w))
    }
}

/// Format a Duration for display, choosing the best unit.
fn fmt_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos < 1_000 {
        format!("{nanos}ns")
    } else if nanos < 1_000_000 {
        format!("{:.3}µs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.3}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}

/// Large pool of emojis for random fallback — visually distinct & fun.
const EMOJI_POOL: &[&str] = &[
    "🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "⚫", "⚪", "🟤", "🍎", "🍊", "🍋", "🍏", "🍇", "🌰", "🥝",
    "🍑", "💎", "🧊", "🪨", "🌀", "🔶", "🔷", "💠", "⭐", "🌟", "💫", "🪐", "🌙", "🌈", "🔮", "🎯",
    "🎪", "🎭", "🎲", "🎸", "🥁", "🎺", "🎻", "🎵", "🧬", "🔬", "🧪", "🧫", "🔭", "💡", "🔋", "🦊",
    "🐼", "🐨", "🦁", "🐯", "🐻", "🦄", "🐸", "🦉", "🤖", "👾", "🎃", "😼", "👻", "👽", "🧳", "🧶",
    "🐶", "🐱", "🐭", "🐹", "🐰", "🐮", "🐷", "🐵", "🐔", "🐧", "🐦", "🐤", "🦆", "🦅", "🌱", "🐾",
    "🌻", "🎋", "🍄", "🌴", "🐎", "🐟", "🐝", "🍅", "🍟", "🍔", "🍕", "🍖", "🍗", "🍩", "🍿", "🍰",
    "🧋", "🍫", "🍬", "🍭", "🍮", "🍯", "🍹", "🎮", "🧩", "🪁", "🎉", "💊", "🦠", "🏮", "🔖", "🔍",
    "📚", "🎈", "📢", "📣", "🃏", "👓", "📦", "🚀",
];

/// Assign an emoji to a task name based on keywords, with random fallback.
fn task_emoji(name: &str) -> &'static str {
    use rand::prelude::*;
    let lower = name.to_lowercase();
    if lower.contains("preprocess") {
        "🔧"
    } else if lower.contains("postprocess") {
        "📦"
    } else if lower.contains("dry-run") || lower.contains("warmup") {
        "🔥"
    } else if lower.contains("inference") || lower.contains("forward") || lower.contains("run") {
        "🧬"
    } else if lower.contains("generate") {
        "🎲"
    } else if lower.contains("encode") {
        "🔑"
    } else if lower.contains("decode") {
        "🔓"
    } else if lower.contains("convert") || lower.contains("->") {
        "🔄"
    } else if lower.contains("annotate") || lower.contains("draw") || lower.contains("render") {
        "🎉"
    } else if lower.contains("init")
        || lower.contains("build")
        || lower.contains("create")
        || lower.contains("new")
    {
        "⛳"
    } else if lower.contains("image") {
        "🌠"
    } else {
        EMOJI_POOL.choose(&mut rand::rng()).copied().unwrap_or("🍊")
    }
}

/// Assign an emoji to a group name.
fn group_emoji(name: &str) -> &'static str {
    let lower = name.to_lowercase();
    if lower.contains("dataloader") || lower.contains("data") || lower.contains("data loader") {
        "🔍"
    } else if lower.contains("annotator") || lower.contains("annotate") {
        "🎨"
    } else if lower.contains("engine") || lower.contains("ort") || lower.contains("onnxruntime") {
        "👾"
    } else {
        "🎉"
    }
}

/// A group with its children, preserving insertion order.
struct Group {
    name: String,
    children: Vec<Child>,
    order: usize,
}

struct Child {
    name: String,
    durations: Vec<Duration>,
    order: usize,
}

impl Child {
    fn total(&self) -> Duration {
        self.durations.iter().sum()
    }
    fn avg(&self) -> Duration {
        let n = self.durations.len();
        if n == 0 {
            Duration::ZERO
        } else {
            self.total() / n as u32
        }
    }
    fn count(&self) -> usize {
        self.durations.len()
    }
}

impl Group {
    fn total(&self) -> Duration {
        self.children.iter().map(|c| c.total()).sum()
    }
}

/// Build groups from raw perf data. Keys use `::` as separator.
/// Single-segment keys become their own group with no children.
fn build_groups(raw: &[(String, Vec<Duration>)]) -> Vec<Group> {
    use std::collections::HashMap;
    let mut group_map: HashMap<String, Group> = HashMap::new();
    let mut group_order: HashMap<String, usize> = HashMap::new();
    let mut next_group = 0usize;

    for (next_child, (key, durations)) in raw.iter().enumerate() {
        let (group_name, child_name) = if let Some(pos) = key.find("::") {
            (key[..pos].to_string(), key[pos + 2..].to_string())
        } else {
            (key.clone(), String::new())
        };

        let gorder = *group_order.entry(group_name.clone()).or_insert_with(|| {
            let o = next_group;
            next_group += 1;
            o
        });

        let group = group_map
            .entry(group_name.clone())
            .or_insert_with(|| Group {
                name: group_name.clone(),
                children: Vec::new(),
                order: gorder,
            });

        if child_name.is_empty() {
            group.children.push(Child {
                name: group_name.clone(),
                durations: durations.clone(),
                order: next_child,
            });
        } else {
            group.children.push(Child {
                name: child_name,
                durations: durations.clone(),
                order: next_child,
            });
        }
    }

    let mut groups: Vec<Group> = group_map.into_values().collect();
    groups.sort_by_key(|g| g.order);
    for g in &mut groups {
        g.children.sort_by_key(|c| c.order);
    }
    groups
}

// Fixed emoji display width used everywhere (2 display columns for emoji + 1 space after)
const EMOJI_DW: usize = 2;

/// Show the performance chart.
///
/// `bar_width` controls the maximum progress-bar length (in characters).
pub fn perf_chart_with_width(bar_width: usize) {
    let raw = perf_collect();
    if raw.is_empty() {
        println!("\n📊 No performance data available");
        return;
    }

    let mut groups = build_groups(&raw);
    if groups.is_empty() {
        return;
    }

    // Synthesize first-level parent entries from sub-tasks when the outer
    // perf! wrapper didn't complete (e.g. early exit / ESC during video decode).
    // e.g. if "decode-video/stream::read" exists but "decode-video/stream" does not,
    // create a synthetic "decode-video/stream" with total = sum of sub-task totals.
    for g in &mut groups {
        let first_level: std::collections::HashSet<&str> = g
            .children
            .iter()
            .filter(|c| !c.name.contains("::"))
            .map(|c| c.name.as_str())
            .collect();

        let mut synthetic: std::collections::HashMap<String, (Duration, usize)> =
            std::collections::HashMap::new();
        for c in &g.children {
            if let Some(pos) = c.name.find("::") {
                let prefix = &c.name[..pos];
                if !first_level.contains(prefix) {
                    let e = synthetic
                        .entry(prefix.to_string())
                        .or_insert((Duration::ZERO, c.order));
                    e.0 += c.total();
                    e.1 = e.1.min(c.order);
                }
            }
        }
        for (name, (total, order)) in synthetic {
            g.children.push(Child {
                name,
                durations: vec![total],
                order,
            });
        }
        g.children.sort_by_key(|c| c.order);
    }

    // For the chart we only show "first-level" children — those whose name
    // does NOT contain "::" (sub-tasks like "decode-video/stream::read" are hidden).
    // Pre-build filtered child index lists per group.
    let chart_children: Vec<Vec<usize>> = groups
        .iter()
        .map(|g| {
            let idxs: Vec<usize> = g
                .children
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.name.contains("::"))
                .map(|(i, _)| i)
                .collect();
            idxs
        })
        .collect();

    // Global max (for group-level bar scaling) — use first-level children totals
    let global_max = groups
        .iter()
        .enumerate()
        .map(|(gi, g)| {
            let total: Duration = chart_children[gi]
                .iter()
                .map(|&ci| g.children[ci].total())
                .sum();
            total
        })
        .max()
        .unwrap_or(Duration::ZERO);

    if global_max == Duration::ZERO {
        return;
    }

    // Compute max label display-width across all groups & first-level children.
    let mut max_dw = 0usize;
    for (gi, g) in groups.iter().enumerate() {
        let header_dw = EMOJI_DW + 1 + g.name.len();
        max_dw = max_dw.max(header_dw);

        let idxs = &chart_children[gi];
        let show_children =
            idxs.len() > 1 || (idxs.len() == 1 && g.children[idxs[0]].name != g.name);
        if show_children {
            for &ci in idxs {
                let child_dw = 4 + EMOJI_DW + 1 + g.children[ci].name.len();
                max_dw = max_dw.max(child_dw);
            }
        }
    }
    let label_dw = max_dw + 1; // +1 for spacing before │

    // Pre-compute stats for alignment: find max time-string width and whether any count > 1
    struct ChildStat {
        time_str: String,
        count: usize,
    }
    let mut all_stats: Vec<Vec<ChildStat>> = Vec::new();
    let mut max_time_dw = 0usize;
    let mut has_any_multi = false;
    for (gi, g) in groups.iter().enumerate() {
        let mut gstats = Vec::new();
        let idxs = &chart_children[gi];
        let show_children =
            idxs.len() > 1 || (idxs.len() == 1 && g.children[idxs[0]].name != g.name);
        if show_children {
            for &ci in idxs {
                let c = &g.children[ci];
                let ts = fmt_duration(c.avg());
                max_time_dw = max_time_dw.max(ts.len());
                if c.count() > 1 {
                    has_any_multi = true;
                }
                gstats.push(ChildStat {
                    time_str: ts,
                    count: c.count(),
                });
            }
        }
        all_stats.push(gstats);
    }

    println!("\n\n🚀 usls Performance Analysis Chart");
    println!("═══════════════════════════════════════════════════════════════\n");

    for (gi, g) in groups.iter().enumerate() {
        let idxs = &chart_children[gi];

        // Group total from first-level children only (avoids double-counting sub-tasks)
        let g_total: Duration = idxs.iter().map(|&ci| g.children[ci].total()).sum();

        // ── group header bar (scaled to global max) ──
        let ratio = g_total.as_nanos() as f64 / global_max.as_nanos() as f64;
        let bar_len = ((ratio * bar_width as f64) as usize).max(3);
        let bar = "█".repeat(bar_len);
        let pad = " ".repeat(bar_width.saturating_sub(bar_len) + 1);

        let ge = fixed_emoji(group_emoji(&g.name), EMOJI_DW);
        let header = format!("{ge} {}", g.name);
        println!(
            "{}│{bar}{pad}{}",
            pad_to(&header, label_dw),
            fmt_duration(g_total),
        );

        // ── first-level children (skip if single child with same name as group) ──
        let show_children =
            idxs.len() > 1 || (idxs.len() == 1 && g.children[idxs[0]].name != g.name);
        if show_children {
            // Per-module scaling: max child avg within first-level children
            let local_max = idxs
                .iter()
                .map(|&ci| g.children[ci].avg())
                .max()
                .unwrap_or(Duration::ZERO);

            for (si, &ci) in idxs.iter().enumerate() {
                let c = &g.children[ci];
                let is_last = si == idxs.len() - 1;
                let branch = if is_last { "└─" } else { "├─" };

                let c_avg = c.avg();
                let c_bar_len = if local_max > Duration::ZERO {
                    let r = c_avg.as_nanos() as f64 / local_max.as_nanos() as f64;
                    ((r * bar_width as f64) as usize).max(2)
                } else {
                    2
                };
                let c_bar = "█".repeat(c_bar_len);
                let c_pad = " ".repeat(bar_width.saturating_sub(c_bar_len) + 1);

                let te = fixed_emoji(task_emoji(&c.name), EMOJI_DW);
                let child_label = format!(" {branch} {te} {}", c.name);

                let stat = &all_stats[gi][si];
                // Left-align time to max_time_dw, then aligned (xN) if any child has count > 1
                let time_padded = format!("{:<w$}", stat.time_str, w = max_time_dw);
                let count_part = if has_any_multi {
                    if stat.count > 1 {
                        format!(" (x{})", stat.count)
                    } else {
                        "     ".to_string() // same width as " (xN)" for alignment
                    }
                } else {
                    String::new()
                };

                println!(
                    "{}│{c_bar}{c_pad}{time_padded}{count_part}",
                    pad_to(&child_label, label_dw),
                );
            }
        }

        // ── scale line (└ must align with │ above, time left-aligned with stats) ──
        let left_pad = " ".repeat(label_dw);
        println!("{left_pad}└{}", "─".repeat(bar_width));
        println!(
            "{left_pad}0{}{}",
            " ".repeat(bar_width + 1),
            fmt_duration(g_total),
        );
        println!();
    }

    println!("💡 Tip: use usls::perf_dashboard() to view more details.\n");
}

/// Show performance chart with default bar width (30).
pub fn perf_chart() {
    perf_chart_with_width(30);
}

/// Show detailed performance dashboard (table format).
pub fn perf_dashboard() {
    let raw = perf_collect();
    if raw.is_empty() {
        println!("\n📊 No performance data available");
        return;
    }

    let groups = build_groups(&raw);

    println!("\n\n🚀 usls Performance Dashboard");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    // Compute column widths using display_width for emoji-aware sizing
    let mut max_name_dw = 0usize;
    for g in &groups {
        for c in &g.children {
            let full = if c.name == g.name {
                format!("{} {}", fixed_emoji(group_emoji(&g.name), EMOJI_DW), c.name)
            } else {
                format!(
                    "{} {}::{}",
                    fixed_emoji(group_emoji(&g.name), EMOJI_DW),
                    g.name,
                    c.name
                )
            };
            max_name_dw = max_name_dw.max(display_width(&full));
        }
        let subtotal = format!("  ({} total)", g.name);
        max_name_dw = max_name_dw.max(display_width(&subtotal));
    }
    let col_name = max_name_dw + 3;
    let col_num = 8;
    let col_time = 14;

    let header_label = pad_to("Task", col_name);
    println!(
        " {header_label}{:<col_num$}{:<col_time$}{:<col_time$}{:<col_time$}{:<col_time$}",
        "Count", "Avg", "Min", "Max", "Total"
    );
    let sep_len = col_name + col_num + col_time * 4 + 2;
    println!(" {}", "─".repeat(sep_len));

    for g in &groups {
        let ge = fixed_emoji(group_emoji(&g.name), EMOJI_DW);

        for c in &g.children {
            let full_name = if c.name == g.name {
                format!("{ge} {}", c.name)
            } else {
                format!("{ge} {}::{}", g.name, c.name)
            };
            let count = c.count();
            let avg = fmt_duration(c.avg());
            let total = fmt_duration(c.total());
            let min = c
                .durations
                .iter()
                .min()
                .map(|d| fmt_duration(*d))
                .unwrap_or_else(|| "-".into());
            let max = c
                .durations
                .iter()
                .max()
                .map(|d| fmt_duration(*d))
                .unwrap_or_else(|| "-".into());

            println!(
                " {}{count:<col_num$}{avg:<col_time$}{min:<col_time$}{max:<col_time$}{total:<col_time$}",
                pad_to(&full_name, col_name),
            );
        }

        // Group subtotal
        let g_total = g.total();
        let subtotal_label = format!("  ({} total)", g.name);
        println!(
            " {}{:<col_num$}{:<col_time$}{:<col_time$}{:<col_time$}{:<col_time$}",
            pad_to(&subtotal_label, col_name),
            "-",
            "-",
            "-",
            "-",
            fmt_duration(g_total),
        );
        println!();
    }

    // Grand total
    let grand_total: Duration = groups.iter().map(|g| g.total()).sum();
    let grand_count: usize = groups
        .iter()
        .flat_map(|g| g.children.iter())
        .map(|c| c.count())
        .sum();
    println!(" {}", "═".repeat(sep_len));
    println!(
        " {}{:<col_num$}{:<col_time$}{:<col_time$}{:<col_time$}{:<col_time$}",
        pad_to("Grand Total", col_name),
        grand_count,
        "-",
        "-",
        "-",
        fmt_duration(grand_total),
    );
    println!();
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_perf_interface() {
        crate::perf_clear();

        crate::perf!("TEST::op_a", {
            thread::sleep(Duration::from_millis(1));
        });
        crate::perf!("TEST::op_b", {
            thread::sleep(Duration::from_millis(2));
        });

        super::perf_chart();
        super::perf_dashboard();

        crate::perf_clear();
    }

    #[test]
    fn test_display_width() {
        // ASCII
        assert_eq!(super::display_width("hello"), 5);
        // Box drawing
        assert_eq!(super::display_width("├─└│═"), 5);
        // Block element
        assert_eq!(super::display_width("████"), 4);
        // Emoji (2 columns each)
        assert_eq!(super::display_width("🔧"), 2);
        assert_eq!(super::display_width("🧬"), 2);
        // µ (Latin-1 Supplement, 1 column)
        assert_eq!(super::display_width("µ"), 1);
    }

    #[test]
    fn test_fixed_emoji() {
        // All emojis should normalize to exactly EMOJI_DW display columns
        let emojis = ["🔧", "🧬", "⭐", "📦", "🎨", "🤖"];
        for e in emojis {
            let fixed = super::fixed_emoji(e, super::EMOJI_DW);
            assert_eq!(
                super::display_width(&fixed),
                super::EMOJI_DW,
                "emoji {e} fixed_width mismatch"
            );
        }
    }
}
