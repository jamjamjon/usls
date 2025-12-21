## Quick Start

```shell
cargo run -r -F vlm --example smolvlm -- --scale 500m --source "images/green-car.jpg" --prompt "What's in it?"
cargo run -r -F vlm --example smolvlm -- --scale 500m --source "images/green-car.jpg" --prompt "What color is the car?"
cargo run -r -F vlm --example smolvlm -- --scale 500m --source "images/slanted-text-number.jpg" --prompt "What are these numbers?"
cargo run -r -F vlm --example smolvlm -- --scale 256m --source "images/Statue-of-Liberty-Island-New-York-Bay.jpg" --prompt "Can you describe this image?"
```