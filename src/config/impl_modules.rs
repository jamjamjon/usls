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
            .ok_or_else(|| anyhow::anyhow!("Module {:?} not configured", id))
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

    /// Set the model file path for the Model module.
    pub fn with_model_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::Model).or_default().file = file.into();
        self
    }

    /// Set the dtype for the Model module.
    pub fn with_model_dtype(mut self, dtype: crate::DType) -> Self {
        self.modules.entry(Module::Model).or_default().dtype = dtype;
        self
    }

    /// Set the device for the Model module.
    pub fn with_model_device(mut self, device: crate::Device) -> Self {
        self.modules.entry(Module::Model).or_default().device = device;
        self
    }

    /// Set the dtype for the TextualEncoder module.
    pub fn with_textual_encoder_dtype(mut self, dtype: crate::DType) -> Self {
        self.modules
            .entry(Module::TextualEncoder)
            .or_default()
            .dtype = dtype;
        self
    }

    /// Set input/output indices for the Model module.
    pub fn with_model_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::Model)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set num_dry_run for the Model module.
    pub fn with_model_num_dry_run(mut self, x: usize) -> Self {
        self.modules.entry(Module::Model).or_default().num_dry_run = x;
        self
    }
    // ===== Encoder module convenience methods =====

    /// Set encoder file.
    pub fn with_encoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::Encoder).or_default().file = file.into();
        self
    }

    /// Set encoder ixx.
    pub fn with_encoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::Encoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set decoder file.
    pub fn with_decoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::Decoder).or_default().file = file.into();
        self
    }

    /// Set decoder ixx.
    pub fn with_decoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::Decoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set visual encoder file.
    pub fn with_visual_encoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::VisualEncoder).or_default().file = file.into();
        self
    }

    /// Set textual encoder file.
    pub fn with_textual_encoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::TextualEncoder).or_default().file = file.into();
        self
    }

    /// Set visual module file.
    pub fn with_visual_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::Visual).or_default().file = file.into();
        self
    }

    /// Set visual module ixx.
    pub fn with_visual_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::Visual)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set textual module file.
    pub fn with_textual_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::Textual).or_default().file = file.into();
        self
    }

    /// Set textual decoder file.
    pub fn with_textual_decoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::TextualDecoder).or_default().file = file.into();
        self
    }

    /// Set textual decoder merged file.
    pub fn with_textual_decoder_merged_file(mut self, file: impl Into<String>) -> Self {
        self.modules
            .entry(Module::TextualDecoderMerged)
            .or_default()
            .file = file.into();
        self
    }

    /// Set visual encoder ixx.
    pub fn with_visual_encoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::VisualEncoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set visual projection file.
    pub fn with_visual_projection_file(mut self, file: impl Into<String>) -> Self {
        self.modules
            .entry(Module::VisualProjection)
            .or_default()
            .file = file.into();
        self
    }

    /// Set visual encoder batch min opt max.
    pub fn with_visual_encoder_batch_min_opt_max(
        mut self,
        min: usize,
        opt: usize,
        max: usize,
    ) -> Self {
        self.modules
            .entry(Module::VisualEncoder)
            .or_default()
            .iiixs
            .insert(
                0,
                crate::Iiix {
                    i: 0,
                    ii: 0,
                    x: crate::MinOptMax::from((min, opt, max)),
                },
            );
        self
    }

    /// Set textual encoder batch min opt max.
    pub fn with_textual_encoder_batch_min_opt_max(
        mut self,
        min: usize,
        opt: usize,
        max: usize,
    ) -> Self {
        self.modules
            .entry(Module::TextualEncoder)
            .or_default()
            .iiixs
            .insert(
                0,
                crate::Iiix {
                    i: 0,
                    ii: 0,
                    x: crate::MinOptMax::from((min, opt, max)),
                },
            );
        self
    }

    /// Set encoder batch min opt max.
    pub fn with_encoder_batch_min_opt_max(mut self, min: usize, opt: usize, max: usize) -> Self {
        self.modules
            .entry(Module::Encoder)
            .or_default()
            .iiixs
            .insert(
                0,
                crate::Iiix {
                    i: 0,
                    ii: 0,
                    x: crate::MinOptMax::from((min, opt, max)),
                },
            );
        self
    }

    /// Set decoder batch min opt max.
    pub fn with_decoder_batch_min_opt_max(mut self, min: usize, opt: usize, max: usize) -> Self {
        self.modules
            .entry(Module::Decoder)
            .or_default()
            .iiixs
            .insert(
                0,
                crate::Iiix {
                    i: 0,
                    ii: 0,
                    x: crate::MinOptMax::from((min, opt, max)),
                },
            );
        self
    }

    /// Set size encoder ixx.
    pub fn with_size_encoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::SizeEncoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set size decoder ixx.
    pub fn with_size_decoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::SizeDecoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set coord encoder ixx.
    pub fn with_coord_encoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::CoordEncoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set coord decoder ixx.
    pub fn with_coord_decoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::CoordDecoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set coord encoder file.
    pub fn with_coord_encoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::CoordEncoder).or_default().file = file.into();
        self
    }

    /// Set coord decoder file.
    pub fn with_coord_decoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::CoordDecoder).or_default().file = file.into();
        self
    }

    /// Set size encoder file.
    pub fn with_size_encoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::SizeEncoder).or_default().file = file.into();
        self
    }

    /// Set size decoder file.
    pub fn with_size_decoder_file(mut self, file: impl Into<String>) -> Self {
        self.modules.entry(Module::SizeDecoder).or_default().file = file.into();
        self
    }

    /// Set visual projection ixx.
    pub fn with_visual_projection_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::VisualProjection)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set textual encoder ixx.
    pub fn with_textual_encoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::TextualEncoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }

    /// Set textual decoder ixx.
    pub fn with_textual_decoder_ixx(
        mut self,
        io: usize,
        idx: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(Module::TextualDecoder)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i: io,
                ii: idx,
                x: value.into(),
            });
        self
    }
}
