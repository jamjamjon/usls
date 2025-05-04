#[derive(aksr::Builder, Clone, PartialEq)]
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
