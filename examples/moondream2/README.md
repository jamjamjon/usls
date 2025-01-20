## Quick Start

```shell
cargo run -r -F cuda --example moondream2 -- --device 'cuda:0' --dtype i8  --scale 2b --task vqa:"What's in this image?"
cargo run -r -F cuda --example moondream2 -- --device 'cuda:0' --dtype i8  --scale 2b --task cap:0
cargo run -r -F cuda --example moondream2 -- --device 'cuda:0' --dtype i8  --scale 2b --task cap:1
cargo run -r -F cuda --example moondream2 -- --device 'cuda:0' --dtype i8  --scale 2b --task open-od:person
cargo run -r -F cuda --example moondream2 -- --device 'cuda:0' --dtype i8  --scale 2b --task open-kpt:person
```

