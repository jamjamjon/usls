# TensorView and TensorViewMut Manual

## Overview

This manual covers the usage of `TensorView` and `TensorViewMut` in the USLS tensor library. These types provide zero-copy views into tensor data, enabling efficient memory usage and high-performance operations without data copying.

## Key Features

- **Zero-Copy Operations**: Views reference existing data without copying
- **Memory Efficient**: Minimal overhead for large tensor operations
- **Chainable**: Multiple view operations can be chained together
- **Type Safe**: Compile-time guarantees for view operations
- **Tuple Syntax**: Clean, readable slicing syntax using tuples

## Method Summary

### Tensor 方法
- `slice()` → 返回 `TensorView<'_>` (只读视图)
- `slice_mut()` → 返回 `TensorViewMut<'_>` (可变视图)

### TensorView 方法
- `slice()` → 返回 `TensorView<'_>` (只读视图)

### TensorViewMut 方法
- `slice()` → 返回 `TensorView<'_>` (只读视图)
- `slice_mut()` → 返回 `TensorViewMut<'_>` (可变视图)

## TensorView - 不可变视图

### 基本用法

```rust
use anyhow::Result;
use usls::tensor::Tensor;
use std::ops::Range;

fn basic_tensor_view() -> Result<()> {
    let tensor = Tensor::rand([4, 4, 3], usls::DType::Fp32)?;
    
    // 方法1: 使用 .view() 创建整个张量的不可变视图
    let full_view = tensor.view();
    println!("Full view shape: {:?}", full_view.shape());
    
    // 方法2: 使用 .slice() 创建部分张量的不可变视图
    let ranges = vec![0..2, 0..2, 0..3];
    let partial_view = tensor.slice(&ranges)?;
    
    println!("Original shape: {:?}", tensor.shape());
    println!("Partial view shape: {:?}", partial_view.shape());
    
    // 视图属性
    println!("View dtype: {:?}", partial_view.dtype());
    println!("View ndim: {}", partial_view.ndim());
    println!("View len: {}", partial_view.len());
    println!("View is_empty: {}", partial_view.is_empty());
    
    Ok(())
}
```

### 视图操作

```rust
fn tensor_view_operations() -> Result<()> {
    let tensor = Tensor::zeros(vec![8, 6, 4]);
    let ranges = vec![2..6, 1..5, 0..4];
    let view = tensor.slice(&ranges)?;
    
    // 转换为拥有的张量
    let owned_tensor = view.to_owned()?
    println!("Converted tensor shape: {:?}", owned_tensor.shape());
    
    // 获取 F32 数据切片（如果类型匹配）
    if let Ok(data_slice) = view.as_f32_slice() {
        println!("F32 data length: {}", data_slice.len());
    }
    
    // 进一步切片
    let sub_view = view.slice((1..3, .., 1..3))?;
    println!("Sub-view shape: {:?}", sub_view.shape());
    
    Ok(())
}
```

## TensorViewMut - 可变视图

### 基本用法

```rust
fn basic_tensor_view_mut() -> Result<()> {
    let mut tensor = Tensor::zeros(vec![4, 4, 3]);
    
    // 方法1: 使用 .view_mut() 创建整个张量的可变视图
    let mut full_view = tensor.view_mut();
    println!("Full mutable view shape: {:?}", full_view.shape());
    
    // 可以对整个视图进行操作
    full_view.fill(1.0)?;
    
    // 方法2: 使用 .slice_mut() 创建部分张量的可变视图
    let ranges = vec![1..3, 1..3, 0..3];
    let mut partial_view = tensor.slice_mut(&ranges)?;
    
    println!("Partial mutable view shape: {:?}", partial_view.shape());
    println!("Mutable view dtype: {:?}", partial_view.dtype());
    
    // 方法3: 使用 .slice() 创建只读视图（从可变视图）
    let readonly_view = partial_view.slice((0..2, 0..2, 0..3))?;
    println!("Readonly view from mutable: {:?}", readonly_view.shape());
    
    // 转换为拥有的张量
    let owned = partial_view.to_owned()?
    println!("Owned tensor shape: {:?}", owned.shape());
    
    Ok(())
}
```

### 可变操作

```rust
fn tensor_view_mut_operations() -> Result<()> {
    let mut tensor = Tensor::rand([6, 8, 4], usls::DType::Fp32)?;
    
    // 创建可变视图
    let ranges = vec![1..5, 2..6, 0..4];
    let mut view = tensor.slice_mut(&ranges)?;
    
    // 可变切片操作
    let sub_tensor = view.slice_mut((1..3, .., 1..3))?;
    println!("Mutable sub-slice shape: {:?}", sub_tensor.shape());
    
    Ok(())
}
```

## 使用示例

### 图像处理示例

```rust
fn image_processing_example() -> Result<()> {
    // 模拟批量图像数据 [batch, height, width, channels]
    let mut batch_images = Tensor::rand([8, 224, 224, 3], usls::DType::Fp32)?;
    
    // 选择单个图像的视图
    let ranges = vec![0..1, 0..224, 0..224, 0..3];
    let single_image_view = batch_images.slice(&ranges)?;
    println!("Single image view: {:?}", single_image_view.shape());
    
    // 中心裁剪
    let crop_ranges = vec![0..8, 50..174, 50..174, 0..3];
    let cropped_view = batch_images.slice(&crop_ranges)?;
    println!("Cropped view: {:?}", cropped_view.shape());
    
    // 通道选择（只保留红色通道）
    let red_channel_ranges = vec![0..8, 0..224, 0..224, 0..1];
    let red_channel_view = batch_images.slice(&red_channel_ranges)?;
    println!("Red channel view: {:?}", red_channel_view.shape());
    
    Ok(())
}
```

### KV 缓存示例

```rust
fn kv_cache_example() -> Result<()> {
    // 模拟 KV 缓存 [batch, num_heads, seq_len, head_dim]
    let mut kv_cache = Tensor::zeros(vec![4, 12, 512, 64]);
    
    // 获取特定头的缓存视图
    let head_ranges = vec![0..4, 0..1, 0..512, 0..64];
    let single_head_view = kv_cache.slice(&head_ranges)?;
    println!("Single head cache: {:?}", single_head_view.shape());
    
    // 获取序列的一部分
    let seq_ranges = vec![0..4, 0..12, 0..256, 0..64];
    let partial_seq_view = kv_cache.slice(&seq_ranges)?;
    println!("Partial sequence cache: {:?}", partial_seq_view.shape());
    
    // 可变视图用于更新缓存
    let update_ranges = vec![0..1, 0..12, 100..200, 0..64];
    let mut cache_update_view = kv_cache.slice_mut(&update_ranges)?;
    println!("Cache update view: {:?}", cache_update_view.shape());
    
    Ok(())
}
```

### 序列处理示例

```rust
fn sequence_processing_example() -> Result<()> {
    // 模拟序列数据 [batch, seq_len, hidden_dim]
    let sequence_data = Tensor::rand([16, 512, 768], usls::DType::Fp32)?;
    
    // 截断序列
    let truncate_ranges = vec![0..16, 0..256, 0..768];
    let truncated_view = sequence_data.slice(&truncate_ranges)?;
    println!("Truncated sequence: {:?}", truncated_view.shape());
    
    // 选择特定批次
    let batch_ranges = vec![0..4, 0..512, 0..768];
    let batch_view = sequence_data.slice(&batch_ranges)?;
    println!("Batch subset: {:?}", batch_view.shape());
    
    // 特征维度切片
    let feature_ranges = vec![0..16, 0..512, 0..384];
    let feature_view = sequence_data.slice(&feature_ranges)?;
    println!("Feature subset: {:?}", feature_view.shape());
    
    Ok(())
}
```

## 高级用法

### 视图链式操作

```rust
fn chained_view_operations() -> Result<()> {
    let tensor = Tensor::rand([8, 8, 8, 8], usls::DType::Fp32)?;
    
    // 创建初始视图
    let ranges1 = vec![1..7, 0..8, 0..8, 0..8];
    let view1 = tensor.slice(&ranges1)?;
    
    // 在视图基础上创建子视图
    let ranges2 = vec![0..4, 2..6, 0..8, 0..8];
    let view2 = view1.slice(&ranges2)?;
    
    // 继续切片
    let ranges3 = vec![0..2, 0..4, 1..7, 2..6];
    let final_view = view2.slice(&ranges3)?;
    
    println!("Final chained view: {:?}", final_view.shape());
    
    Ok(())
}
```

### 批处理操作

```rust
fn batch_processing() -> Result<()> {
    let batch_tensor = Tensor::rand([32, 128, 128, 3], usls::DType::Fp32)?;
    
    // 处理每个批次
    for i in 0..32 {
        let batch_ranges = vec![i..i+1, 0..128, 0..128, 0..3];
        let batch_view = batch_tensor.slice(&batch_ranges)?;
        
        // 对单个批次进行操作
        println!("Processing batch {}: {:?}", i, batch_view.shape());
        
        // 可以进一步切片或处理
        let center_crop_ranges = vec![0..1, 32..96, 32..96, 0..3];
        let cropped = batch_view.slice(&center_crop_ranges)?;
        println!("  Cropped: {:?}", cropped.shape());
    }
    
    Ok(())
}
```

## 最佳实践

### 1. 内存效率 🚀

```rust
// ✅ 好：使用视图避免数据复制
let view = tensor.slice(&ranges)?;
let result = process_view(&view)?;

// ❌ 避免：不必要的数据复制
let copied_tensor = view.to_owned()?
let result = process_tensor(&copied_tensor)?;
```

### 2. 生命周期管理 🔒

```rust
// ✅ 好：确保原始张量的生命周期
fn process_with_view(tensor: &Tensor) -> Result<()> {
    let view = tensor.slice(&ranges)?;
    // 在这里使用 view
    Ok(())
}

// ❌ 避免：视图超出原始张量生命周期
// fn bad_example() -> TensorView {
//     let tensor = Tensor::zeros(vec![4, 4]);
//     tensor.slice(&ranges).unwrap() // 错误：tensor 被销毁
// }
```

### 3. 错误处理 🛡️

```rust
// ✅ 好：适当的错误处理
fn safe_slicing(tensor: &Tensor) -> Result<()> {
    let ranges = vec![0..2, 0..2];
    
    match tensor.slice(&ranges) {
        Ok(view) => {
            println!("View created: {:?}", view.shape());
            Ok(())
        }
        Err(e) => {
            eprintln!("Slicing failed: {}", e);
            Err(e)
        }
    }
}
```

### 4. 性能优化 ⚡

```rust
// ✅ 好：连续内存访问
let contiguous_ranges = vec![0..100, 0..200]; // 连续切片
let view = tensor.slice(&contiguous_ranges)?;

// ✅ 好：避免小切片的频繁创建
let batch_size = 32;
for i in (0..total_size).step_by(batch_size) {
    let end = (i + batch_size).min(total_size);
    let batch_ranges = vec![i..end, 0..dim_size];
    let batch_view = tensor.slice(&batch_ranges)?;
    // 处理批次
}
```

## 性能优化

### 内存布局考虑

```rust
fn memory_layout_optimization() -> Result<()> {
    let tensor = Tensor::rand([1000, 1000], usls::DType::Fp32)?;
    
    // ✅ 好：行优先访问（连续内存）
    for i in 0..1000 {
        let row_ranges = vec![i..i+1, 0..1000];
        let row_view = tensor.slice(&row_ranges)?;
        // 处理行数据
    }
    
    // ⚠️ 注意：列访问可能不够高效
    for j in 0..1000 {
        let col_ranges = vec![0..1000, j..j+1];
        let col_view = tensor.slice(&col_ranges)?;
        // 处理列数据（可能涉及跨步访问）
    }
    
    Ok(())
}
```

### 批处理优化

```rust
fn batch_optimization() -> Result<()> {
    let large_tensor = Tensor::rand([10000, 512], usls::DType::Fp32)?;
    
    // ✅ 好：合理的批次大小
    let batch_size = 256;
    for start in (0..10000).step_by(batch_size) {
        let end = (start + batch_size).min(10000);
        let batch_ranges = vec![start..end, 0..512];
        let batch_view = large_tensor.slice(&batch_ranges)?;
        
        // 批量处理，提高缓存效率
        process_batch(&batch_view)?;
    }
    
    Ok(())
}

fn process_batch(batch: &usls::TensorView) -> Result<()> {
    // 批量处理逻辑
    println!("Processing batch of shape: {:?}", batch.shape());
    Ok(())
}
```

## 总结

TensorView 和 TensorViewMut 提供了强大的零拷贝视图功能：

- 🚀 **高性能**: 零拷贝操作，避免不必要的内存分配
- 🔒 **内存安全**: Rust 借用检查器保证安全性
- 🎯 **灵活性**: 支持复杂的切片和视图操作
- ⚡ **效率**: 适合性能关键的应用场景
- 🛡️ **可靠性**: 编译时类型检查和错误处理

通过合理使用 TensorView 和 TensorViewMut，可以构建高效、安全的张量处理应用！ 🦀✨