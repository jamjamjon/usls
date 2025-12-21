use crate::ResizeFilter;

#[derive(Clone, Copy)]
pub struct Bound {
    pub start: u32,
    pub size: u32,
}

#[derive(Clone, Default)]
pub struct Coefficients {
    pub values: Vec<f64>,
    pub window_size: usize,
    pub bounds: Vec<Bound>,
}

pub fn precompute_coefficients(
    in_size: u32,
    in0: f64,
    in1: f64,
    out_size: u32,
    filter: fn(f64) -> f64,
    filter_support: f64,
    adaptive_kernel_size: bool,
) -> Coefficients {
    if in_size == 0 || out_size == 0 {
        return Coefficients::default();
    }
    let scale = (in1 - in0) / out_size as f64;
    if scale <= 0.0 {
        return Coefficients::default();
    }

    let filter_scale = if adaptive_kernel_size {
        scale.max(1.0)
    } else {
        1.0
    };
    let filter_radius = filter_support * filter_scale;
    let window_size = filter_radius.ceil() as usize * 2 + 1;

    let recip_filter_scale = 1.0 / filter_scale;
    let count_of_coeffs = window_size * out_size as usize;

    let mut coeffs: Vec<f64> = Vec::with_capacity(count_of_coeffs);
    let mut bounds: Vec<Bound> = Vec::with_capacity(out_size as usize);

    for out_x in 0..out_size {
        let in_center = in0 + (out_x as f64 + 0.5) * scale;
        let x_min = (in_center - filter_radius).floor().max(0.0) as u32;
        let x_max = (in_center + filter_radius).ceil().min(in_size as f64) as u32;

        let cur_index = coeffs.len();
        let mut ww: f64 = 0.0;

        let center = in_center - 0.5;
        let mut bound_start = x_min;
        let mut bound_end = x_max;

        for x in x_min..x_max {
            let w: f64 = filter((x as f64 - center) * recip_filter_scale);
            if x == bound_start && w == 0.0 {
                bound_start += 1;
            } else {
                coeffs.push(w);
                ww += w;
            }
        }

        for &c in coeffs.iter().rev() {
            if bound_end <= bound_start || c != 0.0 {
                break;
            }
            bound_end -= 1;
        }

        if ww != 0.0 {
            coeffs[cur_index..].iter_mut().for_each(|w| *w /= ww);
        }

        coeffs.resize(cur_index + window_size, 0.0);
        bounds.push(Bound {
            start: bound_start,
            size: bound_end - bound_start,
        });
    }

    Coefficients {
        values: coeffs,
        window_size,
        bounds,
    }
}

fn box_filter(x: f64) -> f64 {
    if x > -0.5 && x <= 0.5 {
        1.0
    } else {
        0.0
    }
}

fn bilinear_filter(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

fn hamming_filter(x: f64) -> f64 {
    use std::f64::consts::PI;
    let x = x.abs();
    if x == 0.0 {
        1.0
    } else if x >= 1.0 {
        0.0
    } else {
        let x = x * PI;
        (0.54 + 0.46 * x.cos()) * x.sin() / x
    }
}

fn catmul_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn mitchell_filter(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (7.0 * x / 6.0 - 2.0) * x * x + 16.0 / 18.0
    } else if x < 2.0 {
        ((2.0 - 7.0 * x / 18.0) * x - 10.0 / 3.0) * x + 16.0 / 9.0
    } else {
        0.0
    }
}

fn gaussian(x: f64, r: f64) -> f64 {
    use std::f64::consts::PI;
    ((2.0 * PI).sqrt() * r).recip() * (-x.powi(2) / (2.0 * r.powi(2))).exp()
}

fn gaussian_filter(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        gaussian(x, 0.5)
    } else {
        0.0
    }
}

fn sinc_filter(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x == 0.0 {
        1.0
    } else {
        let x = x * PI;
        x.sin() / x
    }
}

fn lanczos_filter(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc_filter(x) * sinc_filter(x / 3.0)
    } else {
        0.0
    }
}

#[allow(clippy::type_complexity)]
pub fn get_filter_func(filter: ResizeFilter) -> (fn(f64) -> f64, f64) {
    match filter {
        ResizeFilter::Box => (box_filter as fn(f64) -> f64, 0.5),
        ResizeFilter::Bilinear => (bilinear_filter as fn(f64) -> f64, 1.0),
        ResizeFilter::Hamming => (hamming_filter as fn(f64) -> f64, 1.0),
        ResizeFilter::CatmullRom => (catmul_filter as fn(f64) -> f64, 2.0),
        ResizeFilter::Mitchell => (mitchell_filter as fn(f64) -> f64, 2.0),
        ResizeFilter::Gaussian => (gaussian_filter as fn(f64) -> f64, 3.0),
        ResizeFilter::Lanczos3 => (lanczos_filter as fn(f64) -> f64, 3.0),
    }
}

pub struct Normalizer16 {
    precision: u8,
    starts: Vec<i32>,
    sizes: Vec<i32>,
    offsets: Vec<i32>,
    coeffs: Vec<i16>,
}

impl Normalizer16 {
    pub fn new(coefficients: Coefficients) -> Self {
        let max_weight = coefficients
            .values
            .iter()
            .max_by(|&x, &y| x.partial_cmp(y).unwrap())
            .unwrap_or(&0.0)
            .to_owned();

        let precision_bits: u8 = 32 - 8 - 2;
        let max_coeffs_precision: u8 = 16 - 1;

        let mut precision = 0u8;
        for cur_precision in 0..precision_bits {
            precision = cur_precision;
            let next_value: i32 = (max_weight * (1i32 << (precision + 1)) as f64).round() as i32;
            if next_value >= (1i32 << max_coeffs_precision) {
                break;
            }
        }
        if precision < 4 {
            precision = 4;
        }

        let mut starts: Vec<i32> = Vec::with_capacity(coefficients.bounds.len());
        let mut sizes: Vec<i32> = Vec::with_capacity(coefficients.bounds.len());
        let mut offsets: Vec<i32> = Vec::with_capacity(coefficients.bounds.len());
        let mut coeffs: Vec<i16> = Vec::new();

        let scale = (1i64 << precision) as f64;
        let mut base: usize = 0;
        if coefficients.window_size > 0 {
            let coef_chunks = coefficients.values.chunks_exact(coefficients.window_size);
            for (chunk, bound) in coef_chunks.zip(&coefficients.bounds) {
                starts.push(bound.start as i32);
                sizes.push(bound.size as i32);
                offsets.push(base as i32);

                let take_n = bound.size as usize;
                for &v in chunk.iter().take(take_n) {
                    coeffs.push((v * scale).round() as i16);
                    base += 1;
                }
            }
        }

        Self {
            precision,
            starts,
            sizes,
            offsets,
            coeffs,
        }
    }

    pub fn precision(&self) -> u8 {
        self.precision
    }

    pub fn starts(&self) -> &[i32] {
        &self.starts
    }

    pub fn sizes(&self) -> &[i32] {
        &self.sizes
    }

    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    pub fn coeffs(&self) -> &[i16] {
        &self.coeffs
    }
}

pub fn compute_convolution_1d(
    in_size: u32,
    out_size: u32,
    filter: ResizeFilter,
    adaptive_kernel_size: bool,
) -> Normalizer16 {
    let (filter_fn, filter_support) = get_filter_func(filter);
    let coeffs = precompute_coefficients(
        in_size,
        0.0,
        in_size as f64,
        out_size,
        filter_fn,
        filter_support,
        adaptive_kernel_size,
    );
    Normalizer16::new(coeffs)
}
