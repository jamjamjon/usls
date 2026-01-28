# Module System

**usls** uses a flexible module-based architecture, allowing models to be composed of multiple ONNX components. This is especially important for Vision-Language Models (VLMs) and complex encoder-decoder pipelines.

## 🧩 The `Module` Enum

All configurable components in **usls** are identified by the `Module` enum:

```rust
pub enum Module {
    // Standard vision models
    Model,

    // Vision-Language components
    Visual,
    Textual,

    // Encoder-Decoder pipelines
    Encoder,
    Decoder,
    VisualEncoder,
    TextualEncoder,
    VisualDecoder,
    TextualDecoder,
    
    // Projections & specialized layers
    VisualProjection,
    TextualProjection,
    
    // Custom modules for extensibility
    Custom(String),
}
```

## Why use Modules?

1.  **Granular Control**: You can place different modules on different devices (e.g., `VisualEncoder` on GPU, `TextualDecoder` on CPU).
2.  **Mixed Precision**: Set `DType::Fp16` for performance-critical modules while keeping others at `DType::Fp32`.
3.  **Optimization**: Configure specific execution provider settings per module.
