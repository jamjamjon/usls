mod display;
mod r#impl;

pub use display::{perf_chart, perf_chart_with_width, perf_dashboard};
pub use r#impl::{__perf_record, perf_clear};
