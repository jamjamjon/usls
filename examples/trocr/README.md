## Quick Start

```shell
cargo run -r -F vlm -F cuda --example trocr -- --device cuda --dtype fp16 --scale s --kind printed

cargo run -r -F vlm -F cuda --example trocr -- --device cuda --dtype fp16 --scale s --kind hand-written

```


```shell
Ys([Y { Texts: [Text("PLEASE LOWER YOUR VOLUME")] }, Y { Texts: [Text("HELLO RUST")] }])
```