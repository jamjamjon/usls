## Quick Start

```shell
cargo run -r --example db
```

### Speed test

| Model           | Image size | TensorRT<br />f16<br />batch=1<br />(ms) | TensorRT<br />f32<br />batch=1<br />(ms) | CUDA<br />f32<br />batch=1<br />(ms) |
| --------------- | ---------- | ---------------------------------------- | ---------------------------------------- | ------------------------------------ |
| ppocr-v3-db-dyn | 640x640    | 1.8585                                   | 2.5739                                   | 4.3314                               |
| ppocr-v4-db-dyn | 640x640    | 2.0507                                   | 2.8264                                   | 6.6064                               |

***Test on RTX3060***

## Results

![](https://github.com/jamjamjon/assets/releases/download/db/demo-paper.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-sign.png)
