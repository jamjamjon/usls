#![cfg(all(feature = "cuda-runtime", feature = "ort-load-dynamic"))]

#[cfg(feature = "ort-download-binaries")]
compile_error!(
    "cuda_pad_consistency must be run with --no-default-features to disable ort-download-binaries"
);

use usls::{
    CpuTransformExecutor, CropMode, CudaPreprocessor, Image, ImagePlan, ImageTensorLayout,
    ImageTransform, PadFillMode, PadMode, TransformExecutor,
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

fn assert_vec_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(x.to_bits() == y.to_bits(), "mismatch at {i}: {x} vs {y}");
    }
}

fn run_plan_cpu_cuda(img: Image, plan: &ImagePlan) -> (Vec<f32>, Vec<f32>) {
    static CPU: OnceLock<CpuTransformExecutor> = OnceLock::new();
    static CUDA: OnceLock<Mutex<CudaPreprocessor>> = OnceLock::new();

    let cpu_executor = CPU.get_or_init(CpuTransformExecutor::new);
    let cuda_executor = CUDA.get_or_init(|| Mutex::new(CudaPreprocessor::new(0).unwrap()));

    let cpu = cpu_executor
        .execute_plan(&[img.clone()], plan)
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
            .execute_plan(&[img], plan)
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

#[test]
fn cpu_cuda_pad_to_multiple_wrap_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::ToMultiple {
            window_size: 8,
            fill_mode: PadFillMode::Wrap,
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let cpu_executor = CpuTransformExecutor::new();
    let cuda_executor = CudaPreprocessor::new(0).unwrap();

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

    let cuda = cuda_executor
        .execute_plan(&[img], &plan)
        .unwrap()
        .0
        .as_host()
        .unwrap()
        .0
        .iter()
        .copied()
        .collect::<Vec<_>>();

    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_pad_to_multiple_constant_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::ToMultiple {
            window_size: 8,
            fill_mode: PadFillMode::Constant(7),
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_pad_fixed_constant_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::Fixed {
            top: 3,
            bottom: 5,
            left: 7,
            right: 2,
            fill_mode: PadFillMode::Constant(11),
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_pad_fixed_reflect_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::Fixed {
            top: 3,
            bottom: 5,
            left: 7,
            right: 2,
            fill_mode: PadFillMode::Reflect,
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_pad_fixed_replicate_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::Fixed {
            top: 3,
            bottom: 5,
            left: 7,
            right: 2,
            fill_mode: PadFillMode::Replicate,
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_pad_fixed_wrap_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::Fixed {
            top: 3,
            bottom: 5,
            left: 7,
            right: 2,
            fill_mode: PadFillMode::Wrap,
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_pad_to_multiple_reflect_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::ToMultiple {
            window_size: 8,
            fill_mode: PadFillMode::Reflect,
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_pad_to_multiple_replicate_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Pad(PadMode::ToMultiple {
            window_size: 8,
            fill_mode: PadFillMode::Replicate,
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}

#[test]
fn cpu_cuda_crop_center_consistency() {
    if !should_run_cuda_tests() {
        return;
    }

    let img = make_test_image(23, 17);

    let plan = ImagePlan {
        transforms: vec![ImageTransform::Crop(CropMode::Center {
            width: 31,
            height: 29,
        })],
        layout: ImageTensorLayout::NHWC,
        normalize: false,
        mean: None,
        std: None,
        unsigned: false,
        pad_image: false,
        pad_size: 8,
        do_resize: false,
        resize_mode_type: None,
    };

    let (cpu, cuda) = run_plan_cpu_cuda(img, &plan);
    assert_vec_eq(&cpu, &cuda);
}
