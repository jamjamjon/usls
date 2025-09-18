## Quick Start

```shell
# single image
cargo run -r --example fastvlm -- --source ./assets/bus.jpg

# batch inference
cargo run -r --example fastvlm -- --prompt "Describe the image in detail." --max-tokens 512 --source ./assets/bus.jpg --source ./assets/dog.jpg --source ./assets/cat.png
```