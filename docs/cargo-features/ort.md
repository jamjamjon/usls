# ONNX Runtime
ONNX Runtime configuration and API version management.

## Configuration

| Feature | Description | Default |
|---------|-------------|:-------:|
| `ort-download-binaries` | Auto-download ONNX Runtime binaries from [pyke](https://ort.pyke.io) | ✓ |
| `ort-load-dynamic` | Manual linking for custom builds. See [Linking Guide](https://ort.pyke.io/setup/linking) | x |

### API Version Selection

This library supports ONNX Runtime versions 1.17 through 1.24 via API version features.

| Feature | ONNX Runtime | Requirements |
|---------|--------------|--------------|
| `ort-api-17` | v1.17 | Baseline |
| `ort-api-18` | v1.18 | - |
| `ort-api-19` | v1.19 | - |
| `ort-api-20` | v1.20 | Adapter API available |
| `ort-api-21` | v1.21 | - |
| `ort-api-22` | v1.22 | - |
| `ort-api-23` | v1.23 | - |
| `ort-api-24` | v1.24 | **Default** - Latest features |

!!! tip "API Version Selection"
    ```toml
    # Default uses api-24 (latest)
    usls = { version = "0.2", features = ["vision"] }
    
    # Specify API version explicitly
    usls = { version = "0.2", features = ["vision", "ort-api-20"] }
    ```

!!! note "Version Compatibility"
    - Each API version includes all features from previous versions
    - Check [ORT multiversion docs](https://ort.pyke.io/setup/multiversion) for minimum version requirements