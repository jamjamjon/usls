#[macro_export]
macro_rules! impl_text_processor_config_methods {
    ($ty:ty, $field:ident) => {
        impl $ty {
            pub fn with_model_max_length(mut self, model_max_length: u64) -> Self {
                self.$field = self.$field.with_model_max_length(model_max_length);
                self
            }
            pub fn with_tokenizer_file(mut self, tokenizer_file: &str) -> Self {
                self.$field = self.$field.with_tokenizer_file(tokenizer_file);
                self
            }
            pub fn with_config_file(mut self, config_file: &str) -> Self {
                self.$field = self.$field.with_config_file(config_file);
                self
            }
            pub fn with_special_tokens_map_file(mut self, special_tokens_map_file: &str) -> Self {
                self.$field = self
                    .$field
                    .with_special_tokens_map_file(special_tokens_map_file);
                self
            }
            pub fn with_tokenizer_config_file(mut self, tokenizer_config_file: &str) -> Self {
                self.$field = self
                    .$field
                    .with_tokenizer_config_file(tokenizer_config_file);
                self
            }
            pub fn with_generation_config_file(mut self, generation_config_file: &str) -> Self {
                self.$field = self
                    .$field
                    .with_generation_config_file(generation_config_file);
                self
            }
            pub fn with_vocab_file(mut self, vocab_file: &str) -> Self {
                self.$field = self.$field.with_vocab_file(vocab_file);
                self
            }
            pub fn with_temperature(mut self, temperature: f32) -> Self {
                self.$field = self.$field.with_temperature(temperature);
                self
            }
            pub fn with_topp(mut self, topp: f32) -> Self {
                self.$field = self.$field.with_topp(topp);
                self
            }
        }
    };
}
