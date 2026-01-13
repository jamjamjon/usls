use crate::{Config, Module, ORTConfig};

impl Config {
    /// Add an module to the config.
    pub fn with_module(mut self, id: Module, config: ORTConfig) -> Self {
        self.modules.insert(id, config);
        self
    }

    /// Get a reference to an module config.
    pub fn get_module(&self, id: &Module) -> Option<&ORTConfig> {
        self.modules.get(id)
    }

    /// Get a mutable reference to an module config.
    pub fn get_module_mut(&mut self, id: &Module) -> Option<&mut ORTConfig> {
        self.modules.get_mut(id)
    }

    /// Take (consume) an module config, removing it from the map.
    pub fn take_module(&mut self, id: &Module) -> anyhow::Result<ORTConfig> {
        self.modules
            .remove(id)
            .ok_or_else(|| anyhow::anyhow!("Module {id:?} not configured"))
    }

    /// Convenience method to configure Model module with a fluent builder.
    pub fn with_model(self, config: ORTConfig) -> Self {
        self.with_module(Module::Model, config)
    }

    /// Convenience method to configure Visual module with a fluent builder.
    pub fn with_visual(self, config: ORTConfig) -> Self {
        self.with_module(Module::Visual, config)
    }

    /// Convenience method to configure Textual module with a fluent builder.
    pub fn with_textual(self, config: ORTConfig) -> Self {
        self.with_module(Module::Textual, config)
    }

    /// Convenience method to configure VisualEncoder module with a fluent builder.
    pub fn with_visual_encoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::VisualEncoder, config)
    }

    /// Convenience method to configure TextualEncoder module with a fluent builder.
    pub fn with_textual_encoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::TextualEncoder, config)
    }

    /// Convenience method to configure Encoder module with a fluent builder.
    pub fn with_encoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::Encoder, config)
    }

    /// Convenience method to configure Decoder module with a fluent builder.
    pub fn with_decoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::Decoder, config)
    }

    /// Convenience method to configure TextualDecoder module with a fluent builder.
    pub fn with_textual_decoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::TextualDecoder, config)
    }

    /// Convenience method to configure TextualDecoderMerged module with a fluent builder.
    pub fn with_textual_decoder_merged(self, config: ORTConfig) -> Self {
        self.with_module(Module::TextualDecoderMerged, config)
    }

    /// Convenience method to configure VisualProjection module with a fluent builder.
    pub fn with_visual_projection(self, config: ORTConfig) -> Self {
        self.with_module(Module::VisualProjection, config)
    }

    /// Convenience method to configure SizeEncoder module with a fluent builder.
    pub fn with_size_encoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::SizeEncoder, config)
    }

    /// Convenience method to configure SizeDecoder module with a fluent builder.
    pub fn with_size_decoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::SizeDecoder, config)
    }

    /// Convenience method to configure CoordEncoder module with a fluent builder.
    pub fn with_coord_encoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::CoordEncoder, config)
    }

    /// Convenience method to configure CoordDecoder module with a fluent builder.
    pub fn with_coord_decoder(self, config: ORTConfig) -> Self {
        self.with_module(Module::CoordDecoder, config)
    }
}
