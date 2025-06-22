# 🦀 USLS Tensor 切片操作手册

这份手册详细介绍了 USLS 库中 Tensor 切片操作的统一 API 和语法。

## 📋 目录

1. [概述](#概述)
2. [核心 API](#核心-api)
3. [切片语法](#切片语法)
4. [使用示例](#使用示例)
5. [高级用法](#高级用法)
6. [最佳实践](#最佳实践)

## 概述

USLS 提供了统一的 Tensor 切片方式，使用现代化的 `s()` 方法，支持原生 Rust 切片语法。为了简化 API，我们移除了数组语法支持，仅保留更灵活的元组语法。

### 主要特性

- 🎯 **直观语法**: 支持 `..`, `..4`, `1..5`, `3` 等原生 Rust 切片语法
- 🚀 **高性能**: 零开销抽象，编译时优化
- 🛡️ **类型安全**: 编译时检查切片规范
- 🧹 **简洁 API**: 统一使用 `s()` 方法，移除冗余接口

## 核心 API

### `tensor.slice()` - 统一切片方法 ⭐

唯一推荐的切片方法，支持原生 Rust 语法：

```rust
use usls::tensor::Tensor;

let tensor = Tensor::zeros(vec![4, 4, 3]);

// 使用元组语法（唯一支持的语法）
let slice = tensor.slice((.., ..2, 1))?;
let slice = tensor.slice((0..2, .., 1..3))?;
let slice = tensor.slice((2.., 1, ..))?;
```

**注意**: 为了 API 简洁性，已移除数组语法支持。只支持元组语法。

## 切片语法

### SliceOrIndex 枚举类型

```rust
pub enum SliceOrIndex {
    Range(Range<usize>),     // 范围切片: 0..5
    RangeFrom(usize),        // 从某处开始: 2..
    FullSlice,               // 完整切片: ..
    Index(usize),            // 单个索引: 3
}
```

### 支持的语法形式

| 语法 | 含义 | SliceOrIndex 等价 |
|------|------|------------------|
| `..` | 完整切片 | `SliceOrIndex::FullSlice` |
| `..4` | 从开始到索引4 | `SliceOrIndex::Range(0..4)` |
| `2..` | 从索引2到结束 | `SliceOrIndex::RangeFrom(2)` |
| `1..5` | 从索引1到索引5 | `SliceOrIndex::Range(1..5)` |
| `3` | 单个索引3 | `SliceOrIndex::Index(3)` |

## 使用示例

### 基础示例

```rust
use anyhow::Result;
use usls::tensor::Tensor;

fn basic_examples() -> Result<()> {
    let tensor = Tensor::rand([4, 4, 3], usls::DType::Fp32)?;
    println!("Original shape: {:?}", tensor.shape()); // [4, 4, 3]
    
    // 1. 完整切片
    let slice1 = tensor.slice((.., .., ..))?;
    println!("Full slice: {:?}", slice1.shape()); // [4, 4, 3]
    
    // 2. 索引切片
    let slice2 = tensor.slice((0, .., ..))?;
    println!("Index slice: {:?}", slice2.shape()); // [4, 3]
    
    // 3. 范围切片
    let slice3 = tensor.slice((.., 1..3, ..))?;
    println!("Range slice: {:?}", slice3.shape()); // [4, 2, 3]
    
    // 4. 混合切片
    let slice4 = tensor.slice((0, 1..3, 2))?;
    println!("Mixed slice: {:?}", slice4.shape()); // [2]
    
    // 5. 范围从语法
    let slice5 = tensor.slice((2.., ..2, 1))?;
    println!("Range from slice: {:?}", slice5.shape()); // [2, 2]
    
    Ok(())
}
```

### 元组语法优势

```rust
// ✅ 元组语法 - 支持混合类型，灵活强大
let slice1 = tensor.slice((.., ..2, 1))?;           // 完整 + 范围 + 索引
let slice2 = tensor.slice((0, .., 1..3))?;          // 索引 + 完整 + 范围
let slice3 = tensor.slice((2.., 1, ..))?;           // 范围从 + 索引 + 完整
```

### 多维张量示例

```rust
fn multidimensional_examples() -> Result<()> {
    // 2D 张量
    let tensor_2d = Tensor::zeros(vec![10, 5]);
    let slice_2d = tensor_2d.slice((2..8, ..3))?;
    println!("2D slice: {:?}", slice_2d.shape()); // [6, 3]
    
    // 4D 张量
    let tensor_4d = Tensor::zeros(vec![5, 4, 3, 2]);
    let slice_4d = tensor_4d.slice((0..2, 1..3, 0..2, 0))?;
    println!("4D slice: {:?}", slice_4d.shape()); // [2, 2, 2]
    
    // 高维张量
    let tensor_6d = Tensor::zeros(vec![8, 7, 6, 5, 4, 3]);
    let slice_6d = tensor_6d.slice((1.., ..5, 2, .., 1..3, ..))?;
    println!("6D slice: {:?}", slice_6d.shape()); // [7, 5, 5, 2, 3]
    
    Ok(())
}
```

## 高级用法

### 链式切片

```rust
fn chained_slicing() -> Result<()> {
    let tensor = Tensor::rand([8, 6, 4], usls::DType::Fp32)?;
    
    // 链式切片操作
    let result = tensor
        .slice((0..6, .., ..))?     // 第一次切片: [6, 6, 4]
    .slice((.., 1..5, ..))?     // 第二次切片: [6, 4, 4]
    .slice((2..4, .., 1..3))?;  // 第三次切片: [2, 4, 2]
    
    println!("Chained result: {:?}", result.shape());
    Ok(())
}
```

### 条件切片

```rust
fn conditional_slicing(use_full_slice: bool) -> Result<()> {
    let tensor = Tensor::zeros(vec![8, 8]);
    
    let slice_spec = if use_full_slice {
        (.., ..)
    } else {
        (2..6, 1..7)
    };
    
    let result = tensor.slice(slice_spec)?;
    println!("Conditional slice: {:?}", result.shape());
    
    Ok(())
}
```

### 复杂切片模式

```rust
fn complex_patterns() -> Result<()> {
    let tensor = Tensor::zeros(vec![10, 8, 6, 4]);
    
    // 跳跃式切片
    let skip_slice = tensor.slice((1..9, 0..6, .., 1..3))?;
    println!("Skip slice: {:?}", skip_slice.shape()); // [8, 6, 6, 2]
    
    // 边界切片
    let boundary = tensor.slice((..2, 6.., 0, ..))?;
    println!("Boundary slice: {:?}", boundary.shape()); // [2, 2, 4]
    
    // 中心切片
    let center = tensor.slice((2..8, 1..7, 1..5, 1..3))?;
    println!("Center slice: {:?}", center.shape()); // [6, 6, 4, 2]
    
    Ok(())
}
```

## 最佳实践

### 1. 统一使用 s() 方法 🎯

```rust
// ✅ 推荐：统一使用 s() 方法
let slice = tensor.slice((.., 1..5, 0))?;

// ❌ 避免：使用已弃用的方法
// let slice = tensor.slice_dyn(&slices)?;  // 已弃用
// let slice = tensor.slice_easy(&spec)?;   // 已移除
```

### 2. 利用元组语法的灵活性 🔧

```rust
// ✅ 好：元组语法支持混合类型
let slice = tensor.slice((0, .., 1..3))?;
let slice = tensor.slice((2.., 1, ..2))?;
let slice = tensor.slice((.., ..4, 2..))?;
```

### 3. 性能考虑 ⚡

```rust
// ✅ 高效：编译时已知的切片规范
let slice = tensor.slice((1..5, .., 2))?;

// ✅ 高效：复用切片规范
let spec = (1..5, .., 2);
let slice1 = tensor1.slice(spec)?;
let slice2 = tensor2.slice(spec)?;
```

### 4. 错误处理 🛡️

```rust
fn safe_slicing() -> Result<()> {
    let tensor = Tensor::zeros(vec![4, 4, 3]);
    
    // 检查维度边界
    let shape = tensor.shape();
    if shape[1] >= 2 {
        let slice = tensor.slice((.., ..2, ..))?;
        println!("Safe slice: {:?}", slice.shape());
    }
    
    Ok(())
}
```

### 5. 代码可读性 📖

```rust
// ✅ 清晰：使用有意义的变量名
let batch_slice = tensor.slice((0..batch_size, .., ..))?;
let channel_slice = tensor.slice((.., .., channel_idx))?;

// ✅ 清晰：添加注释说明复杂切片
let feature_map = tensor.slice((
    ..,                        // 保持批次维度
    height_start..height_end,  // 裁剪高度
    width_start..width_end,    // 裁剪宽度
))?;
```

## 迁移指南

如果你之前使用了其他切片方法，这里是迁移指南：

```rust
// 旧代码：使用 slice_dyn
let old_slice = tensor.slice_dyn(&[
    SliceOrIndex::FullSlice,
    SliceOrIndex::Range(0..2),
    SliceOrIndex::Index(1),
])?;

// 新代码：使用 s() 方法
let new_slice = tensor.slice((.., 0..2, 1))?;

// 旧代码：使用数组语法
// let array_slice = tensor.slice([0..2, 1..3, 0..1])?;  // 不再支持

// 新代码：使用元组语法
let tuple_slice = tensor.slice((0..2, 1..3, 0))?;
```

## 总结

USLS 的 Tensor 切片系统现在更加简洁和统一：

- 🚀 **统一 API**: 只需要学习 `tensor.slice()` 一个方法
- 🎯 **灵活语法**: 元组语法支持所有切片需求
- 🧹 **简洁设计**: 移除冗余接口，降低学习成本
- ⚡ **高性能**: 零开销抽象，编译时优化
- 🛡️ **类型安全**: 编译时检查切片规范

享受简洁高效的张量操作体验！ 🦀✨