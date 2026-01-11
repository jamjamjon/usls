## Quick Start


### Depth Anything v1
```
cargo run -F cuda --example depth-estimation -- depth-anything --device cuda --processor-device cuda --dtype f16 --scale s --ver 1 
```

### Depth Anything v2
```
cargo run -F cuda --example depth-estimation -- depth-anything --device cuda --processor-device cuda --dtype f16 --scale s --ver 2 
```

### Depth Anything v3 - Multi-View
```
cargo run -F cuda --example depth-estimation -- depth-anything --device cuda --processor-device cuda --dtype q4f16 --scale s --ver 3 --kind multi
```


### Depth Anything v3 - MONO
```
cargo run -F cuda --example depth-estimation -- depth-anything --device cuda --processor-device cuda --dtype q4f16 --scale l --ver 3 --kind mono
```

### Depth Anything v3 - Metric
```
cargo run -F cuda --example depth-estimation -- depth-anything --device cuda --processor-device cuda --dtype q4f16 --scale l --ver 3 --kind metric
```


### Depth Pro
```
cargo run -F cuda --example depth-estimation -- depth-pro --device cuda --processor-device cuda --dtype q4f16 
```


## Results

![](https://github.com/jamjamjon/assets/releases/download/depth-anything/demo.png)


![](https://github.com/jamjamjon/assets/releases/download/depth-pro/demo-depth-pro.png)
