#![cfg(all(feature = "cuda-runtime", feature = "ort-load-dynamic"))]

#[cfg(feature = "ort-download-binaries")]
compile_error!("cuda_resize_consistency must be run with --no-default-features to disable ort-download-binaries");

use usls::{
    CpuTransformExecutor, CudaPreprocessor, Image, ImagePlan, ImageTensorLayout, ImageTransform,
    ResizeAlg, ResizeFilter, ResizeMode, TransformExecutor,
};

use std::sync::{Mutex, OnceLock};

fn should_run_cuda_tests() -> bool {
    matches!(
        std::env::var("USLS_TEST_CUDA").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

fn make_test_image(width: u32, height: u32) -> Image {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut buf = vec![0u8; (width * height * 3) as usize];
    rng.fill_bytes(&mut buf);
    Image::from_u8s(&buf, width, height).unwrap()
}

fn run_resize_plan_cpu_cuda(img: Image, mode: ResizeMode) -> (Vec<f32>, Vec<f32>) {
    static CPU: OnceLock<CpuTransformExecutor> = OnceLock::new();
    static CUDA: OnceLock<Mutex<CudaPreprocessor>> = OnceLock::new();

    let cpu_executor = CPU.get_or_init(CpuTransformExecutor::new);
    let cuda_executor = CUDA.get_or_init(|| Mutex::new(CudaPreprocessor::new(0).unwrap()));

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Resize(mode)],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: true,
        resize_mode_type: None,
    };

    let cpu = cpu_executor
        .execute_plan(&[img.clone()], &plan)
        .unwrap()
        .0
        .as_host()
        .unwrap()
        .0
        .iter()
        .copied()
        .collect::<Vec<_>>();

    let cuda = {
        let cuda_executor = cuda_executor.lock().unwrap();
        cuda_executor
            .execute_plan(&[img], &plan)
            .unwrap()
            .0
            .as_host()
            .unwrap()
            .0
            .iter()
            .copied()
            .collect::<Vec<_>>()
    };

    (cpu, cuda)
}

fn assert_vec_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(x.to_bits() == y.to_bits(), "mismatch at {i}: {x} vs {y}");
    }
}

#[test]
fn cpu_cuda_resize_nearest_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);
    let out_w = 31;
    let out_h = 29;

    let (cpu, cuda) = run_resize_plan_cpu_cuda(
        img,
        ResizeMode::FitExact {
            width: out_w,
            height: out_h,
            alg: ResizeAlg::Nearest,
        },
    );
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_resize_interpolation_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(19, 27);
    let out_w = 37;
    let out_h = 13;

    let filters = [
        ResizeFilter::Box,
        ResizeFilter::Bilinear,
        ResizeFilter::Hamming,
        ResizeFilter::CatmullRom,
        ResizeFilter::Mitchell,
        ResizeFilter::Gaussian,
        ResizeFilter::Lanczos3,
    ];

    for filter in filters {
        let (cpu, cuda) = run_resize_plan_cpu_cuda(
            img.clone(),
            ResizeMode::FitExact {
                width: out_w,
                height: out_h,
                alg: ResizeAlg::Interpolation(filter),
            },
        );
        assert_vec_eq(&cpu, &cuda);
    }
}

#[test]
fn cpu_cuda_resize_convolution_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(19, 27);
    let out_w = 37;
    let out_h = 13;

    let filters = [
        ResizeFilter::Box,
        ResizeFilter::Bilinear,
        ResizeFilter::Hamming,
        ResizeFilter::CatmullRom,
        ResizeFilter::Mitchell,
        ResizeFilter::Gaussian,
        ResizeFilter::Lanczos3,
    ];

    for filter in filters {
        let (cpu, cuda) = run_resize_plan_cpu_cuda(
            img.clone(),
            ResizeMode::FitExact {
                width: out_w,
                height: out_h,
                alg: ResizeAlg::Convolution(filter),
            },
        );
        assert_vec_eq(&cpu, &cuda);
    }
}
