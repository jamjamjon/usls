// TODO

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

    fn quantize_value(&self, val: f32, bin_size: f64, max_bin: i32) -> i32 {
        ((val as f64 / bin_size).floor() as i32).clamp(0, max_bin - 1)
    }

    fn dequantize_value(&self, val: i32, bin_size: f64) -> f32 {
        ((val as f64 + 0.5) * bin_size) as f32
    }

    fn quantize_internal(&self, input: &[f32], size: (usize, usize)) -> Vec<i32> {
        let (bins_w, bins_h) = self.bins;
        let (size_w, size_h) = size;

        let size_per_bin_w = size_w as f64 / bins_w as f64;
        let size_per_bin_h = size_h as f64 / bins_h as f64;

        match input.len() {
            4 => vec![
                self.quantize_value(input[0], size_per_bin_w, bins_w as i32),
                self.quantize_value(input[1], size_per_bin_h, bins_h as i32),
                self.quantize_value(input[2], size_per_bin_w, bins_w as i32),
                self.quantize_value(input[3], size_per_bin_h, bins_h as i32),
            ],
            2 => vec![
                self.quantize_value(input[0], size_per_bin_w, bins_w as i32),
                self.quantize_value(input[1], size_per_bin_h, bins_h as i32),
            ],
            _ => panic!("Unsupported input length"),
        }
    }

    fn dequantize_internal(&self, input: &[i32], size: (usize, usize)) -> Vec<f32> {
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
            _ => panic!("Unsupported input length"),
        }
    }

    pub fn quantize(&self, input: &[f32], size: (usize, usize)) -> Vec<i32> {
        self.quantize_internal(input, size)
    }

    pub fn dequantize(&self, input: &[i32], size: (usize, usize)) -> Vec<f32> {
        self.dequantize_internal(input, size)
    }
}
