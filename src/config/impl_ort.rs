use crate::{Config, DType, Device, Module};

// ===== Macros for generating ORT configuration functions =====

/// Generate functions for specific modules.
///
/// Pattern: `with_<module>_<field>()`
///
/// # Parameters
/// - `$field`: Configuration field name
/// - `$ty`: Field type
/// - `$setter`: Closure to set the field value
/// - `$module`: Module names (variadic)
macro_rules! impl_ort_for_modules {
    ($field:ident, $ty:ty, $setter:expr, $($module:ident),+) => {
        $(
            paste::paste! {
                #[doc = "Set `" $field "` for the `" $module "` module."]
                pub fn [<with_ $module:snake _ $field>](mut self, x: $ty) -> Self {
                    let config = self.modules
                        .entry(crate::Module::$module)
                        .or_default();
                    $setter(config, x);
                    self
                }
            }
        )+
    };
}

/// Generate function for a specific module parameter.
///
/// Pattern: `with_module_<field>()`
///
/// # Parameters
/// - `$field`: Configuration field name
/// - `$ty`: Field type
/// - `$setter`: Closure to set the field value
macro_rules! impl_ort_for_module {
    ($field:ident, $ty:ty, $setter:expr) => {
        paste::paste! {
            #[doc = "Set `" $field "` for a specific module."]
            pub fn [<with_module_ $field>](mut self, module: crate::Module, x: $ty) -> Self {
                if let Some(config) = self.modules.get_mut(&module) {
                    $setter(config, x);
                }
                self
            }
        }
    };
}

/// Generate function for all modules.
///
/// Pattern: `with_<field>_all()`
///
/// # Parameters
/// - `$field`: Configuration field name
/// - `$ty`: Field type
/// - `$setter`: Closure to set the field value
macro_rules! impl_ort_for_all {
    ($field:ident, $ty:ty, $setter:expr) => {
        paste::paste! {
            #[doc = "Apply `" $field "` to all modules."]
            pub fn [<with_ $field _all>](mut self, x: $ty) -> Self {
                for config in self.modules.values_mut() {
                    $setter(config, x);
                }
                self
            }
        }
    };
}

/// Generate all three patterns for an ORT field.
///
/// This macro combines [`impl_ort_for_modules!`], [`impl_ort_for_module!`], and [`impl_ort_for_all!`]
/// to generate a complete set of configuration functions for a given ORT field.
///
/// # Parameters
/// - `$field`: Configuration field name
/// - `$ty`: Field type
/// - `$setter`: Closure to set the field value
macro_rules! impl_ort_field {
    ($field:ident, $ty:ty, $setter:expr) => {
        impl_ort_for_modules!(
            $field,
            $ty,
            $setter,
            Model,
            Visual,
            Textual,
            Encoder,
            Decoder,
            VisualEncoder,
            TextualEncoder,
            VisualDecoder,
            TextualDecoder,
            TextualDecoderMerged,
            SizeEncoder,
            SizeDecoder,
            CoordEncoder,
            CoordDecoder,
            VisualProjection,
            TextualProjection
        );
        impl_ort_for_module!($field, $ty, $setter);
        impl_ort_for_all!($field, $ty, $setter);
    };
}

/// Generate batch_min_opt_max functions for specific modules.
///
/// Creates functions to set dynamic batch sizing with minimum, optimal, and maximum values.
/// Also generates `batch_size_min_opt_max` aliases for backward compatibility.
///
/// # Parameters
/// - `$module`: Module names (variadic)
macro_rules! impl_batch_min_opt_max_for_modules {
    ($($module:ident),+) => {
        $(
            paste::paste! {
                #[doc = "Set batch size (min, opt, max) for the `" $module "` module."]
                pub fn [<with_ $module:snake _batch_min_opt_max>](
                    mut self,
                    min: usize,
                    opt: usize,
                    max: usize,
                ) -> Self {
                    self.modules
                        .entry(crate::Module::$module)
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

                #[doc = "Set batch size (min, opt, max) for the `" $module "` module (alias)."]
                pub fn [<with_ $module:snake _batch_size_min_opt_max>](
                    self,
                    min: usize,
                    opt: usize,
                    max: usize,
                ) -> Self {
                    self.[<with_ $module:snake _batch_min_opt_max>](min, opt, max)
                }
            }
        )+
    };
}

/// Generate batch functions for specific modules.
///
/// Creates functions to set a single batch size value.
/// Also generates `batch_size` aliases for backward compatibility.
///
/// # Parameters
/// - `$module`: Module names (variadic)
macro_rules! impl_batch_for_modules {
    ($($module:ident),+) => {
        $(
            paste::paste! {
                #[doc = "Set batch size for the `" $module "` module."]
                pub fn [<with_ $module:snake _batch>](mut self, batch: usize) -> Self {
                    self.modules
                        .entry(crate::Module::$module)
                        .or_default()
                        .iiixs
                        .push(crate::Iiix::from((0, 0, batch.into())));
                    self
                }

                #[doc = "Set batch size for the `" $module "` module (alias)."]
                pub fn [<with_ $module:snake _batch_size>](self, batch: usize) -> Self {
                    self.[<with_ $module:snake _batch>](batch)
                }
            }
        )+
    };
}

/// Generate ixx (input/output index) functions for specific modules.
///
/// Creates functions to configure dynamic input/output tensor dimensions.
///
/// # Parameters
/// - `$module`: Module names (variadic)
macro_rules! impl_ixx_for_modules {
    ($($module:ident),+) => {
        $(
            paste::paste! {
                #[doc = "Set input/output indices for the `" $module "` module."]
                pub fn [<with_ $module:snake _ixx>](
                    mut self,
                    i: usize,
                    ii: usize,
                    value: impl Into<crate::MinOptMax>,
                ) -> Self {
                    self.modules
                        .entry(crate::Module::$module)
                        .or_default()
                        .iiixs
                        .push(crate::Iiix {
                            i,
                            ii,
                            x: value.into(),
                        });
                    self
                }
            }
        )+
    };
}

/// Generate file path functions for specific modules.
///
/// Creates functions to set the ONNX model file path for each module.
///
/// # Parameters
/// - `$module`: Module names (variadic)
macro_rules! impl_file_for_modules {
    ($($module:ident),+) => {
        $(
            paste::paste! {
                #[doc = "Set file path for the `" $module "` module."]
                pub fn [<with_ $module:snake _file>](mut self, x: impl Into<String>) -> Self {
                    self.modules
                        .entry(crate::Module::$module)
                        .or_default()
                        .file = x.into();
                    self
                }
            }
        )+
    };
}

impl Config {
    // ---------------- Special: file -----------------------
    impl_file_for_modules!(
        Model,
        Visual,
        Textual,
        Encoder,
        Decoder,
        VisualEncoder,
        TextualEncoder,
        VisualDecoder,
        TextualDecoder,
        TextualDecoderMerged,
        SizeEncoder,
        SizeDecoder,
        CoordEncoder,
        CoordDecoder,
        VisualProjection,
        TextualProjection
    );

    pub fn with_module_file(mut self, module: crate::Module, x: impl Into<String>) -> Self {
        if let Some(config) = self.modules.get_mut(&module) {
            config.file = x.into();
        }
        self
    }

    pub fn with_file_all(mut self, x: impl Into<String>) -> Self {
        let x = x.into();
        for config in self.modules.values_mut() {
            config.file = x.clone();
        }
        self
    }

    // ---------------- Basic Fields -----------------------
    impl_ort_field!(dtype, DType, |config: &mut crate::ORTConfig, x: DType| {
        config.dtype = x;
    });

    impl_ort_field!(
        device,
        Device,
        |config: &mut crate::ORTConfig, x: Device| {
            config.device = x;
        }
    );

    impl_ort_field!(
        num_dry_run,
        usize,
        |config: &mut crate::ORTConfig, x: usize| {
            config.num_dry_run = x;
        }
    );

    impl_ort_field!(
        graph_opt_level,
        u8,
        |config: &mut crate::ORTConfig, x: u8| {
            config.graph_opt_level = Some(x);
        }
    );

    impl_ort_field!(
        num_intra_threads,
        usize,
        |config: &mut crate::ORTConfig, x: usize| {
            config.num_intra_threads = Some(x);
        }
    );

    impl_ort_field!(
        num_inter_threads,
        usize,
        |config: &mut crate::ORTConfig, x: usize| {
            config.num_inter_threads = Some(x);
        }
    );

    // ---------------- Special: batch -----------------------
    impl_batch_for_modules!(
        Model,
        Visual,
        Textual,
        Encoder,
        Decoder,
        VisualEncoder,
        TextualEncoder,
        VisualDecoder,
        TextualDecoder,
        TextualDecoderMerged,
        SizeEncoder,
        SizeDecoder,
        CoordEncoder,
        CoordDecoder,
        VisualProjection,
        TextualProjection
    );

    pub fn with_module_batch(mut self, module: Module, batch: usize) -> Self {
        if let Some(config) = self.modules.get_mut(&module) {
            config.iiixs.push(crate::Iiix::from((0, 0, batch.into())));
        }
        self
    }

    pub fn with_module_batch_size(self, module: Module, batch: usize) -> Self {
        self.with_module_batch(module, batch)
    }

    pub fn with_batch_all(mut self, batch: usize) -> Self {
        for config in self.modules.values_mut() {
            config.iiixs.push(crate::Iiix::from((0, 0, batch.into())));
        }
        self
    }

    pub fn with_batch_size_all(self, batch: usize) -> Self {
        self.with_batch_all(batch)
    }

    // ---------------- Special: batch_min_opt_max -----------------------
    impl_batch_min_opt_max_for_modules!(
        Model,
        Visual,
        Textual,
        Encoder,
        Decoder,
        VisualEncoder,
        TextualEncoder,
        VisualDecoder,
        TextualDecoder,
        TextualDecoderMerged,
        SizeEncoder,
        SizeDecoder,
        CoordEncoder,
        CoordDecoder,
        VisualProjection,
        TextualProjection
    );

    pub fn with_module_batch_min_opt_max(
        mut self,
        module: Module,
        min: usize,
        opt: usize,
        max: usize,
    ) -> Self {
        self.modules.entry(module).or_default().iiixs.insert(
            0,
            crate::Iiix {
                i: 0,
                ii: 0,
                x: crate::MinOptMax::from((min, opt, max)),
            },
        );
        self
    }

    pub fn with_module_batch_size_min_opt_max(
        self,
        module: Module,
        min: usize,
        opt: usize,
        max: usize,
    ) -> Self {
        self.with_module_batch_min_opt_max(module, min, opt, max)
    }

    pub fn with_batch_min_opt_max_all(mut self, min: usize, opt: usize, max: usize) -> Self {
        for config in self.modules.values_mut() {
            config.iiixs.insert(
                0,
                crate::Iiix {
                    i: 0,
                    ii: 0,
                    x: crate::MinOptMax::from((min, opt, max)),
                },
            );
        }
        self
    }

    pub fn with_batch_size_min_opt_max_all(self, min: usize, opt: usize, max: usize) -> Self {
        self.with_batch_min_opt_max_all(min, opt, max)
    }

    // ---------------- Special: ixx -----------------------
    impl_ixx_for_modules!(
        Model,
        Visual,
        Textual,
        Encoder,
        Decoder,
        VisualEncoder,
        TextualEncoder,
        VisualDecoder,
        TextualDecoder,
        TextualDecoderMerged,
        SizeEncoder,
        SizeDecoder,
        CoordEncoder,
        CoordDecoder,
        VisualProjection,
        TextualProjection
    );

    pub fn with_module_ixx(
        mut self,
        module: Module,
        i: usize,
        ii: usize,
        value: impl Into<crate::MinOptMax>,
    ) -> Self {
        self.modules
            .entry(module)
            .or_default()
            .iiixs
            .push(crate::Iiix {
                i,
                ii,
                x: value.into(),
            });
        self
    }

    pub fn with_ixx_all(mut self, i: usize, ii: usize, value: impl Into<crate::MinOptMax>) -> Self {
        let value = value.into();
        for config in self.modules.values_mut() {
            config.iiixs.push(crate::Iiix {
                i,
                ii,
                x: value.clone(),
            });
        }
        self
    }
}
