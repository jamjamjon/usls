## Quick Start

```shell
cargo run -r -F vision --example ram -- --kind ram --dtype bnb4
cargo run -r -F vision --example ram -- --kind ram++ --dtype q8
```

## Results

```shell
Image: Image { Height: 1080, Width: 810, MediaType: Image(Local), Source: Some("./assets/bus.jpg") }
Texts: [Text { text: "蓝色(blue)" }, Text { text: "建筑(building)" }, Text { text: "公交车(bus)" }, Text { text: "公交车站(bus stop)" }, Text { text: "城市公交(city bus)" }, Text { text: "交叉(cross)" }, Text { text: "人(person)" }, Text { text: "男人(man)" }, Text { text: "人行道/硬路面(pavement)" }, Text { text: "路(road)" }, Text { text: "旅游巴士(tour bus)" }, Text { text: "走(walk)" }]
--------------------------------
Image: Image { Height: 1365, Width: 2048, MediaType: Image(Local), Source: Some("./assets/dog.jpg") }
Texts: [Text { text: "棕色(brown)" }, Text { text: "追逐(chase)" }, Text { text: "牧羊犬(sheepdog)" }, Text { text: "威尔士矮脚狗(corgi)" }, Text { text: "狗(dog)" }, Text { text: "田野/场地/野外(field)" }, Text { text: "草(grass)" }, Text { text: "长满草的(grassy)" }, Text { text: "跑(run)" }, Text { text: "白色(white)" }]
--------------------------------
Image: Image { Height: 688, Width: 927, MediaType: Image(Local), Source: Some("./assets/cat.png") }
Texts: [Text { text: "栏杆(balustrade)" }, Text { text: "蓝色(blue)" }, Text { text: "猫(cat)" }, Text { text: "椅子(chair)" }, Text { text: "甲板(deck)" }, Text { text: "桌子/表格(table)" }, Text { text: "眼睛(eye)" }, Text { text: "窗台(ledge)" }, Text { text: "野餐桌(picnic table)" }, Text { text: "玄关(porch)" }, Text { text: "栏杆/铁轨(rail)" }, Text { text: "暹罗猫(siamese)" }, Text { text: "坐/放置/坐落(sit)" }, Text { text: "盯着(stare)" }, Text { text: "凳子(stool)" }]
--------------------------------
```
