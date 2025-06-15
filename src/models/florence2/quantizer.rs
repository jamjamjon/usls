#[derive(Debug)]
pub struct Quantizer {
    bins: (usize, usize),
}

impl Default for Quantizer {
    fn default() -> Self {
        Self { bins: (1000, 1000) }
    }
}

impl Quantizer {
    pub fn new(bins: (usize, usize)) -> Self {
        Quantizer { bins }
    }

    fn quantize_value(&self, val: f32, bin_size: f64, max_bin: usize) -> usize {
        ((val as f64 / bin_size).floor() as usize).clamp(0, max_bin - 1)
    }

    fn dequantize_value(&self, val: usize, bin_size: f64) -> f32 {
        ((val as f64 + 0.5) * bin_size) as f32
    }

    fn quantize_internal(&self, input: &[usize], size: (usize, usize)) -> Vec<usize> {
        let (bins_w, bins_h) = self.bins;
        let (size_w, size_h) = size;

        let size_per_bin_w = size_w as f64 / bins_w as f64;
        let size_per_bin_h = size_h as f64 / bins_h as f64;

        match input.len() {
            4 => vec![
                self.quantize_value(input[0] as f32, size_per_bin_w, bins_w),
                self.quantize_value(input[1] as f32, size_per_bin_h, bins_h),
                self.quantize_value(input[2] as f32, size_per_bin_w, bins_w),
                self.quantize_value(input[3] as f32, size_per_bin_h, bins_h),
            ],
            2 => vec![
                self.quantize_value(input[0] as f32, size_per_bin_w, bins_w),
                self.quantize_value(input[1] as f32, size_per_bin_h, bins_h),
            ],
            _ => {
                log::error!(
                    "Error: Unsupported input length: {} in Quantizer. Supported lengths: 2, 4",
                    input.len()
                );
                // Return empty vector as fallback instead of panicking
                Vec::new()
            }
        }
    }

    fn dequantize_internal(&self, input: &[usize], size: (usize, usize)) -> Vec<f32> {
        let (bins_w, bins_h) = self.bins;
        let (size_w, size_h) = size;

        let size_per_bin_w = size_w as f64 / bins_w as f64;
        let size_per_bin_h = size_h as f64 / bins_h as f64;

        match input.len() {
            4 => vec![
                self.dequantize_value(input[0], size_per_bin_w),
                self.dequantize_value(input[1], size_per_bin_h),
                self.dequantize_value(input[2], size_per_bin_w),
                self.dequantize_value(input[3], size_per_bin_h),
            ],
            2 => vec![
                self.dequantize_value(input[0], size_per_bin_w),
                self.dequantize_value(input[1], size_per_bin_h),
            ],
            _ => {
                log::error!(
                    "Error: Unsupported input length: {} in Quantizer. Supported lengths: 2, 4",
                    input.len()
                );
                // Return empty vector as fallback instead of panicking
                Vec::new()
            }
        }
    }

    pub fn quantize(&self, input: &[usize], size: (usize, usize)) -> Vec<usize> {
        self.quantize_internal(input, size)
    }

    pub fn dequantize(&self, input: &[usize], size: (usize, usize)) -> Vec<f32> {
        self.dequantize_internal(input, size)
    }
}
