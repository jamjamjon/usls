## Quick Start

```shell
# single image
cargo run -r -F vlm --example fastvlm -- --source ./assets/bus.jpg

# batch inference
cargo run -r -F vlm --example fastvlm -- --prompt "Describe the image in detail." --max-tokens 512 --source ./assets/bus.jpg --source ./assets/dog.jpg --source ./assets/cat.png
```