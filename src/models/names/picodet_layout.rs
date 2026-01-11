//! PicoDet document layout label sets.

/// PicoDet layout taxonomy with 17 annotated regions (titles, text, figures, tables, etc.).
pub const NAMES_PICODET_LAYOUT_17: [&str; 17] = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
];

/// Simplified 5-class layout set (text, title, list, table, figure).
pub const NAMES_PICODET_LAYOUT_5: [&str; 5] = ["Text", "Title", "List", "Table", "Figure"];

/// Minimal 3-class layout set (image, table, seal).
pub const NAMES_PICODET_LAYOUT_3: [&str; 3] = ["image", "table", "seal"];
