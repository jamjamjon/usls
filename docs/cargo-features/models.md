# Model Categories

| Feature | Category | Models | Dependencies | Default |
|---------|----------|--------|-------------|:-------:|
| ***`vision`*** | Core Vision | Detection, Segmentation, Classification, Pose, OBB | - | ✓ |
| **`vlm`** | Vision-Language | CLIP, BLIP, Florence2, multi-modal | `tokenizers`, `ndarray-npy` | x |
| **`mot`** | Tracking | Multi-Object Tracking utilities | - | x |
| **`all-models`** | All | vision + vlm + mot combined | `tokenizers`, `ndarray-npy` | x |



!!! tip "Recommended Setup"
    ```toml
    # Standard computer vision
    features = ["vision"]
    
    # With vision-language capabilities
    features = ["vision", "vlm"]
    
    # Everything included
    features = ["all-models"]
    ```
