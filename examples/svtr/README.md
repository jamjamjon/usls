## Quick Start

```shell
cargo run -r --example svtr
```

### Speed test

| Model                       | Width | TensorRT<br />f16<br />batch=1<br />(ms) | TensorRT<br />f32<br />batch=1<br />(ms) | CUDA<br />f32<br />batch=1<br />(ms) |
| --------------------------- | :---: | :--------------------------------------: | :--------------------------------------: | :----------------------------------: |
| ppocr-v4-server-svtr-ch-dyn | 1500 |                  4.2116                  |                 13.0013                 |               20.8673               |
| ppocr-v4-svtr-ch-dyn        | 1500 |                  2.0435                  |                  3.1959                  |               10.1750               |
| ppocr-v3-svtr-ch-dyn        | 1500 |                  1.8596                  |                  2.9401                  |                6.8210                |

***Test on RTX3060***

## Results

```shell
["./examples/svtr/images/5.png"]: Some(["are closely jointed. Some examples are illustrated in Fig.7."])
["./examples/svtr/images/6.png"]: Some(["小菊儿胡同71号"])
["./examples/svtr/images/4.png"]: Some(["我在南锣鼓捣猫呢"])
["./examples/svtr/images/1.png"]: Some(["你有这么高速运转的机械进入中国，记住我给出的原理"])
["./examples/svtr/images/2.png"]: Some(["冀B6G000"])
["./examples/svtr/images/9.png"]: Some(["from the background, but also separate text instances which"])
["./examples/svtr/images/8.png"]: Some(["110022345"])
["./examples/svtr/images/3.png"]: Some(["粤A·68688"])
["./examples/svtr/images/7.png"]: Some(["Please lower your volume"])
```