use aksr::Builder;

use crate::{impl_meta_methods, InstanceMeta, Style};

#[derive(Builder, Clone, Default)]
pub struct Text {
    text: String,
    meta: InstanceMeta,
    style: Option<Style>,
}

impl std::fmt::Debug for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // f.debug_struct("Text")
        //     .field("text", &self.text)
        //     .field("id", &self.meta.id())
        //     .field("name", &self.meta.name())
        //     .field("confidence", &self.meta.confidence())
        //     .finish()

        let mut f = f.debug_struct("Text");
        f.field("text", &self.text);
        if let Some(id) = &self.meta.id() {
            f.field("id", id);
        }
        if let Some(name) = &self.meta.name() {
            f.field("name", name);
        }
        if let Some(confidence) = &self.meta.confidence() {
            f.field("confidence", confidence);
        }
        f.finish()
    }
}

impl From<String> for Text {
    fn from(text: String) -> Self {
        Self {
            text,
            ..Default::default()
        }
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        Self {
            text: text.to_string(),
            ..Default::default()
        }
    }
}

impl Text {
    impl_meta_methods!();
}
