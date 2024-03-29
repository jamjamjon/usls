This demo showcases how to use [CLIP](https://github.com/openai/CLIP) to compute similarity between texts and images, which can be employed for image-to-text or text-to-image retrieval tasks.

## Quick Start

```shell
cargo run -r --example clip
```

## Or you can manully


### 1.Donwload CLIP ONNX Model

[clip-b32-visual](https://github.com/jamjamjon/assets/releases/download/v0.0.1/clip-b32-visual.onnx)  
[clip-b32-textual](https://github.com/jamjamjon/assets/releases/download/v0.0.1/clip-b32-textual.onnx)


### 2. Specify the ONNX model path in `main.rs`

```Rust
    // visual
    let options_visual = Options::default()
        .with_model("VISUAL_MODEL")  // <= modify this
        .with_i00((1, 1, 4).into())
        .with_profile(false);

    // textual
    let options_textual = Options::default()
        .with_model("TEXTUAL_MODEL")  // <= modify this
        .with_i00((1, 1, 4).into())
        .with_profile(false);
```

### 3. Then, run

```bash
cargo run -r --example clip
```



## Results

```shell
(82.24775%) ./examples/clip/images/carrot.jpg => 几个胡萝卜 
[0.06708972, 0.0067733657, 0.0019306632, 0.8224775, 0.003044935, 0.083962336, 0.014721389]

(85.56889%) ./examples/clip/images/doll.jpg => There is a doll with red hair and a clock on a table 
[0.0786363, 0.0004783095, 0.00060898095, 0.06286741, 0.0006842306, 0.8556889, 0.0010357979]

(90.03625%) ./examples/clip/images/peoples.jpg => Some people holding wine glasses in a restaurant 
[0.07473288, 0.0027821448, 0.0075673857, 0.010874652, 0.003041679, 0.0006387719, 0.9003625]
```


## TODO

* [ ] TensorRT support for textual model
