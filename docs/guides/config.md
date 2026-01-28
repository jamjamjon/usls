# Config System

The `Config` system in **usls** follows a builder pattern with a strict naming convention, making it easy to discover and use APIs.

## API Naming Convention

API naming follows 2 predictable patterns:

1. Per-Module: 
    - `with_<module_name>_<field_name>(<value>)`
    - `with_module_<field_name>(<module_name>, <value>)`
2. Global: 
    - `with_<field_name>_all(<value>)`

### Common Modules
- `model`: Primary model module.
- `visual`: Visual encoder/decoder in VLMs.
- `textual`: Textual encoder/decoder in VLMs.

### Example Usage

```rust
let config = Config::yolo()
    .with_model_device(Device::Cuda(0))    // Set device for model
    .with_model_dtype(DType::Fp16)         // Set precision for model
    .with_batch_all(8)                     // Set batch size (global)
    .commit()?;
```

## Global vs. Per-Module Config

You can apply settings to all modules at once or target specific ones:

```rust
// Apply to all modules
config.with_device_all(Device::Cuda(0));

// Target specific module
config.with_model_device(Device::Cuda(0));
config.with_module_device(Module::Model, Device::Cuda(0));
```

## Dynamic Shapes (TensorRT)

Configure dynamic axes similar to `trtexec`:

```rust
config.with_model_ixx(0, 0, (1, 1, 8)); // axis 0 (batch): min=1, opt=1, max=8
```
