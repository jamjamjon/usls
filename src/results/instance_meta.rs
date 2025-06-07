#[derive(aksr::Builder, Clone, PartialEq)]
/// Metadata for detection instances including ID, confidence, and name.
pub struct InstanceMeta {
    uid: usize,
    id: Option<usize>,
    confidence: Option<f32>,
    name: Option<String>,
}

impl Default for InstanceMeta {
    fn default() -> Self {
        Self {
            uid: {
                static COUNTER: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(1);
                COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            id: None,
            confidence: None,
            name: None,
        }
    }
}

impl std::fmt::Debug for InstanceMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Meta")
            .field("uid", &self.uid)
            .field("id", &self.id)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl InstanceMeta {
    pub fn label(
        &self,
        show_id: bool,
        show_name: bool,
        show_conf: bool,
        decimal_places: usize,
    ) -> String {
        // Format: #id name confidence. e.g.: #0 Person 0.932
        let mut label = String::new();

        //  id
        if let Some(id) = self.id {
            if show_id {
                label.push('#');
                label.push_str(id.to_string().as_str());
            }
        }

        // name
        if let Some(name) = &self.name {
            if show_name {
                let name = if label.is_empty() {
                    name.to_string()
                } else {
                    format!(" {}", name)
                };
                label.push_str(&name);
            }
        }

        // confidence
        if let Some(confidence) = self.confidence {
            if show_conf {
                if label.is_empty() {
                    label.push_str(&format!("{:.decimal_places$}", confidence));
                } else {
                    label.push_str(&format!(" {:.decimal_places$}", confidence));
                }
            }
        }

        label
    }
}

macro_rules! impl_meta_methods {
    () => {
        pub fn with_uid(mut self, uid: usize) -> Self {
            self.meta = self.meta.with_uid(uid);
            self
        }
        pub fn with_id(mut self, id: usize) -> Self {
            self.meta = self.meta.with_id(id);
            self
        }
        pub fn with_name(mut self, name: &str) -> Self {
            self.meta = self.meta.with_name(name);
            self
        }
        pub fn with_confidence(mut self, confidence: f32) -> Self {
            self.meta = self.meta.with_confidence(confidence);
            self
        }
        pub fn uid(&self) -> usize {
            self.meta.uid()
        }
        pub fn name(&self) -> Option<&str> {
            self.meta.name()
        }
        pub fn confidence(&self) -> Option<f32> {
            self.meta.confidence()
        }
        pub fn id(&self) -> Option<usize> {
            self.meta.id()
        }
    };
}
