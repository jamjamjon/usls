impl crate::Config {
    /// Set tokenizer file path.
    pub fn with_tokenizer_file(mut self, file: impl Into<String>) -> Self {
        self.text_processor.tokenizer_file = Some(file.into());
        self
    }

    /// Set tokenizer config file path.
    pub fn with_tokenizer_config_file(mut self, file: impl Into<String>) -> Self {
        self.text_processor.tokenizer_config_file = Some(file.into());
        self
    }

    /// Set model max length for tokenizer.
    pub fn with_model_max_length(mut self, length: u64) -> Self {
        self.text_processor.model_max_length = Some(length);
        self
    }

    /// Set maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, n: u64) -> Self {
        self.text_processor.max_tokens = Some(n);
        self
    }

    /// Set whether to ignore the end-of-sequence token.
    pub fn with_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.text_processor.ignore_eos = ignore_eos;
        self
    }

    /// Set special tokens map file.
    pub fn with_special_tokens_map_file(mut self, file: impl Into<String>) -> Self {
        self.text_processor.special_tokens_map_file = Some(file.into());
        self
    }

    /// Set config file path (for tokenizer).
    pub fn with_config_file(mut self, file: impl Into<String>) -> Self {
        self.text_processor.config_file = Some(file.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.text_processor.temperature = temperature;
        self
    }

    pub fn with_topp(mut self, topp: f32) -> Self {
        self.text_processor.topp = topp;
        self
    }
}
