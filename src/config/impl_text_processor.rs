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
}
