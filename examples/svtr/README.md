## Quick Start

```shell
cargo run -r -F cuda --example svtr -- --device cuda
```

## Results

```shell
["./examples/svtr/images/license-ch-2.png"]: Ys([Y { Texts: [Text("粤A·68688")] }])
["./examples/svtr/images/license-ch.png"]: Ys([Y { Texts: [Text("冀B6G000")] }])
["./examples/svtr/images/sign-ch-2.png"]: Ys([Y { Texts: [Text("我在南锣鼓捣猫呢")] }])
["./examples/svtr/images/sign-ch.png"]: Ys([Y { Texts: [Text("小菊儿胡同71号")] }])
["./examples/svtr/images/text-110022345.png"]: Ys([Y { Texts: [Text("110022345")] }])
["./examples/svtr/images/text-ch.png"]: Ys([Y { Texts: [Text("你有这么高速运转的机械进入中国，记住我给出的原理")] }])
["./examples/svtr/images/text-en-2.png"]: Ys([Y { Texts: [Text("from the background, but also separate text instances which")] }])
["./examples/svtr/images/text-en-dark.png"]: Ys([Y { Texts: [Text("Please lower your volume")] }])
["./examples/svtr/images/text-en.png"]: Ys([Y { Texts: [Text("are closely jointed. Some examples are illustrated in Fig.7.")] }])
["./examples/svtr/images/text-hello-rust-handwritten.png"]: Ys([Y { Texts: [Text("HeloRuSt")] }])

```