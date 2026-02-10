use indicatif::{ProgressBar, ProgressStyle};

/// Standard prefix length for progress bar formatting.
pub(crate) const PREFIX_LENGTH: usize = 12;

/// Progress bar components for building layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PBComponent {
    /// Prefix (e.g., "Downloading").
    Prefix,
    FinishedPrefix,
    /// Message (e.g., filename).
    Message,
    /// Progress bar (e.g., [#####---]).
    Bar,
    /// Decimal current position with commas.
    HumanPos,
    /// Decimal total length with commas.
    HumanLen,
    /// Percentage with 3 decimal places.
    PercentPrecise,
    /// Current position in decimal bytes (e.g., 100 MB).
    DecimalBytes,
    /// Total size in decimal bytes (e.g., 200 MB).
    DecimalTotalBytes,
    /// Current position in binary bytes (e.g., 100 MiB).
    BinaryBytes,
    /// Total size in binary bytes (e.g., 200 MiB).
    BinaryTotalBytes,
    /// Speed in steps per second.
    Speed,
    /// Speed in decimal bytes per second.
    DecimalSpeed,
    /// Speed in binary bytes per second.
    BinarySpeed,
    /// Task spinner.
    Spinner,
    /// Elapsed time (e.g., 42s).
    Elapsed,
    /// Elapsed time (HH:MM:SS).
    ElapsedPrecise,
    /// In Elapsed time (e.g., in 42s).
    InElapsed,
    /// In Elapsed time (in HH:MM:SS).
    InElapsedPrecise,
    /// Remaining time (e.g., 42s).
    ETA,
    /// Remaining time (HH:MM:SS).
    ETAPrecise,
    /// ForHumanLen style: "for {human_len} items"
    ForHumanLen,
    /// X-multiplier style: "x{human_len}"
    XHumanLen,
    /// Binary counter: [MiB/MiB]
    BinaryCounter,
    /// Decimal counter: [MB/MB]
    DecimalCounter,
    /// Decimal progress (e.g., 125/313).
    Counter,
    /// Progress bar with infinity support (e.g., 12/∞).
    CounterWithInfinity,
}

impl std::fmt::Display for PBComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Prefix => write!(f, "{{prefix:>{PREFIX_LENGTH}.cyan.bold}}"),
            Self::FinishedPrefix => write!(f, "{{prefix:>{PREFIX_LENGTH}.green.bold}}"),
            Self::Message => write!(f, "{{msg}}"),
            Self::Bar => write!(f, "|{{bar}}|"),
            Self::HumanPos => write!(f, "{{human_pos}}"),
            Self::HumanLen => write!(f, "{{human_len}}"),
            Self::Counter => write!(f, "{{human_pos}}/{{human_len}}"),
            Self::CounterWithInfinity => write!(f, "{{human_pos}}/∞"),
            Self::PercentPrecise => write!(f, "[{{percent_precise}}%]"),
            Self::DecimalBytes => write!(f, "{{decimal_bytes}}"),
            Self::DecimalTotalBytes => write!(f, "{{decimal_total_bytes}}"),
            Self::BinaryBytes => write!(f, "{{binary_bytes}}"),
            Self::BinaryTotalBytes => write!(f, "{{binary_total_bytes}}"),
            Self::Speed => write!(f, "({{per_sec}})"),
            Self::DecimalSpeed => write!(f, "({{decimal_bytes_per_sec}})"),
            Self::BinarySpeed => write!(f, "({{binary_bytes_per_sec}})"),
            Self::Spinner => write!(f, "{{spinner:.green}}"),
            Self::Elapsed => write!(f, "{{elapsed}}"),
            Self::ElapsedPrecise => write!(f, "{{elapsed_precise}}"),
            Self::InElapsed => write!(f, "in {{elapsed}}"),
            Self::InElapsedPrecise => write!(f, "in {{elapsed_precise}}"),
            Self::ETA => write!(f, "{{eta}}"),
            Self::ETAPrecise => write!(f, "{{eta_precise}}"),
            Self::ForHumanLen => write!(f, "for {{human_len}} iterations"),
            Self::XHumanLen => write!(f, "x{{human_len}}"),
            Self::BinaryCounter => write!(f, "[{{binary_bytes}}/{{binary_total_bytes}}]"),
            Self::DecimalCounter => write!(f, "[{{decimal_bytes}}/{{decimal_total_bytes}}]"),
        }
    }
}

#[derive(Clone)]
pub struct PB {
    inner: ProgressBar,
    layout: Vec<PBComponent>,
    completion_layout: Vec<PBComponent>,
    prefix: &'static str,
    completion_prefix: &'static str,
}

impl Default for PB {
    fn default() -> Self {
        Self {
            inner: ProgressBar::new(0),
            layout: vec![
                PBComponent::Prefix,
                PBComponent::Message,
                PBComponent::Bar,
                PBComponent::Counter,
                PBComponent::Speed,
            ],
            completion_layout: vec![
                PBComponent::FinishedPrefix,
                PBComponent::Message,
                PBComponent::XHumanLen,
                PBComponent::Elapsed,
            ],
            prefix: "",
            completion_prefix: "",
        }
    }
}

impl std::fmt::Debug for PB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PB")
            .field("prefix", &self.prefix)
            .field("completion_prefix", &self.completion_prefix)
            .finish()
    }
}

impl PB {
    pub fn new(total: u64, prefix: &'static str, completion_prefix: &'static str) -> Self {
        let pb = Self {
            inner: ProgressBar::new(total),
            prefix,
            completion_prefix,
            ..Default::default()
        };
        pb.apply_style(false);
        pb
    }

    pub fn fetch(total: u64) -> Self {
        Self::new(total, "Fetching", "Fetched")
            .with_layout(vec![
                PBComponent::Prefix,
                PBComponent::Message,
                PBComponent::Bar,
                PBComponent::PercentPrecise,
                PBComponent::BinaryCounter,
                PBComponent::BinarySpeed,
            ])
            .with_completion_layout(vec![
                PBComponent::FinishedPrefix,
                PBComponent::Message,
                PBComponent::BinaryTotalBytes,
                PBComponent::InElapsed,
            ])
    }

    pub fn fetch_stream() -> Self {
        let pb = Self {
            inner: ProgressBar::new_spinner(),
            prefix: "Fetching",
            completion_prefix: "Fetched",
            ..Default::default()
        };
        pb.apply_style(false);
        pb.with_layout(vec![
            PBComponent::Prefix,
            PBComponent::Message,
            PBComponent::BinaryBytes,
            PBComponent::BinarySpeed,
        ])
        .with_completion_layout(vec![
            PBComponent::FinishedPrefix,
            PBComponent::Message,
            PBComponent::BinaryBytes,
            PBComponent::InElapsed,
        ])
    }

    pub fn iterating(total: u64) -> Self {
        let counter = if total == u64::MAX {
            PBComponent::CounterWithInfinity
        } else {
            PBComponent::Counter
        };

        Self::new(total, "Iterating", "Iterated")
            .with_layout(vec![
                PBComponent::Prefix,
                PBComponent::Message,
                PBComponent::Bar,
                counter,
                PBComponent::Speed,
            ])
            .with_completion_layout(vec![
                PBComponent::FinishedPrefix,
                PBComponent::Message,
                PBComponent::XHumanLen,
                PBComponent::InElapsed,
                PBComponent::Speed,
            ])
    }

    pub fn dry_run(total: u64) -> Self {
        Self::new(total, "DryRun", "DryRun")
            .with_layout(vec![
                PBComponent::Prefix,
                PBComponent::Message,
                PBComponent::Bar,
                PBComponent::Counter,
                PBComponent::Speed,
            ])
            .with_completion_layout(vec![
                PBComponent::FinishedPrefix,
                PBComponent::Message,
                PBComponent::ForHumanLen,
                PBComponent::InElapsed,
            ])
    }

    pub fn with_prefix(mut self, prefix: &'static str) -> Self {
        self.prefix = prefix;
        self.apply_style(false);
        self
    }

    pub fn with_completion_prefix(mut self, prefix: &'static str) -> Self {
        self.completion_prefix = prefix;
        self.apply_style(false);
        self
    }

    pub fn with_layout(mut self, layout: Vec<PBComponent>) -> Self {
        self.layout = layout;
        self.apply_style(false);
        self
    }

    pub fn with_completion_layout(mut self, layout: Vec<PBComponent>) -> Self {
        self.completion_layout = layout;
        self.apply_style(false);
        self
    }

    pub fn with_message(self, msg: &str) -> Self {
        self.inner.set_message(msg.to_string());
        self.apply_style(false);
        self
    }

    pub fn inc(&self, n: u64) {
        self.inner.inc(n);
    }

    pub fn tick(&self) {
        self.inner.tick();
    }

    pub fn set_message(&self, msg: &str) {
        self.inner.set_message(msg.to_string());
        self.apply_style(false);
    }

    pub fn finish(&self, msg: Option<&str>) {
        if let Some(m) = msg {
            self.inner.set_message(m.to_string());
        }
        self.apply_style(true);
        self.inner.finish();
    }

    pub fn inner(&self) -> &ProgressBar {
        &self.inner
    }

    fn apply_style(&self, completed: bool) {
        let template = self.build_template(completed);
        let style = ProgressStyle::with_template(&template).unwrap();
        self.inner.set_style(if completed {
            style
        } else {
            style.progress_chars("██ ")
        });

        let prefix = if completed {
            self.completion_prefix
        } else {
            self.prefix
        };
        self.inner.set_prefix(format!("{prefix:>PREFIX_LENGTH$}"));
    }

    fn build_template(&self, completed: bool) -> String {
        let layout = if completed {
            &self.completion_layout
        } else {
            &self.layout
        };
        let has_message = !self.inner.message().is_empty();

        let estimated_capacity = layout.len() * 20;
        let mut template = String::with_capacity(estimated_capacity);

        for comp in layout {
            if *comp == PBComponent::Message && !has_message {
                continue;
            }

            if !template.is_empty() {
                template.push(' ');
            }

            use std::fmt::Write;
            let _ = write!(&mut template, "{comp}");
        }

        template
    }
}
