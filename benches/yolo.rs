use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use usls::{models::YOLO, DataLoader, Options, Vision, YOLOTask, YOLOVersion};

enum Stage {
    Pre,
    Run,
    Post,
    Pipeline,
}

fn yolo_stage_bench(
    model: &mut YOLO,
    x: &[image::DynamicImage],
    stage: Stage,
    n: u64,
) -> std::time::Duration {
    let mut t_pre = std::time::Duration::new(0, 0);
    let mut t_run = std::time::Duration::new(0, 0);
    let mut t_post = std::time::Duration::new(0, 0);
    let mut t_pipeline = std::time::Duration::new(0, 0);
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let xs = model.preprocess(x).unwrap();
        t_pre += t0.elapsed();

        let t = std::time::Instant::now();
        let xs = model.inference(xs).unwrap();
        t_run += t.elapsed();

        let t = std::time::Instant::now();
        let _ys = black_box(model.postprocess(xs, x).unwrap());
        t_post += t.elapsed();
        t_pipeline += t0.elapsed();
    }
    match stage {
        Stage::Pre => t_pre,
        Stage::Run => t_run,
        Stage::Post => t_post,
        Stage::Pipeline => t_pipeline,
    }
}

pub fn benchmark_cuda(c: &mut Criterion, h: isize, w: isize) -> Result<()> {
    let mut group = c.benchmark_group(format!("YOLO ({}-{})", w, h));
    group
        .significance_level(0.05)
        .sample_size(80)
        .measurement_time(std::time::Duration::new(20, 0));

    let options = Options::default()
        .with_yolo_version(YOLOVersion::V8) // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
        .with_yolo_task(YOLOTask::Detect) // YOLOTask: Classify, Detect, Pose, Segment, Obb
        .with_model("yolo/v8-m-dyn.onnx")?
        .with_cuda(0)
        // .with_cpu()
        .with_num_dry_run(0)
        .with_ixx(0, 2, (320, h, 1280).into())
        .with_ixx(0, 3, (320, w, 1280).into())
        .with_confs(&[0.2, 0.15]);
    let mut model = YOLO::new(options)?;

    let xs = [DataLoader::try_read("./assets/bus.jpg")?];

    group.bench_function("pre-process", |b| {
        b.iter_custom(|n| yolo_stage_bench(&mut model, &xs, Stage::Pre, n))
    });

    group.bench_function("run", |b| {
        b.iter_custom(|n| yolo_stage_bench(&mut model, &xs, Stage::Run, n))
    });

    group.bench_function("post-process", |b| {
        b.iter_custom(|n| yolo_stage_bench(&mut model, &xs, Stage::Post, n))
    });

    group.bench_function("pipeline", |b| {
        b.iter_custom(|n| yolo_stage_bench(&mut model, &xs, Stage::Pipeline, n))
    });

    group.finish();
    Ok(())
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // benchmark_cuda(c, 416, 416).unwrap();
    benchmark_cuda(c, 640, 640).unwrap();
    benchmark_cuda(c, 448, 768).unwrap();
    // benchmark_cuda(c, 800, 800).unwrap();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
