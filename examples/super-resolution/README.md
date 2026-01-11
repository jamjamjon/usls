## Quick Start


### APISR

```shell
cargo run -F cuda --example super-resolution  apisr   --device cuda:0 --processor-device cuda:0 
```

### Swin2SR

```shell
cargo run -F cuda --example super-resolution  swin2sr   --device cuda:0 --processor-device cuda:0 
```

# Results

|   Model   |    Image   |  
| :---------------: | :---------: |
| Original |![](https://github.com/jamjamjon/assets/releases/download/images/ekko.jpg) |
| RRDB-2x | ![](https://github.com/jamjamjon/assets/releases/download/apisr/demo-RRDB-2x.png) |
| GRL-4x |![](https://github.com/jamjamjon/assets/releases/download/apisr/demo-GRL-4x.png) |






