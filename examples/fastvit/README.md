## Quick Start

```shell
cargo run -r -F cuda --example mobileone -- --device cuda --dtype fp16
```
 

```shell
0: Y { Probs:  { Top5: [(263, 0.6109131, Some("Pembroke, Pembroke Welsh corgi")), (264, 0.2062352, Some("Cardigan, Cardigan Welsh corgi")), (231, 0.028572788, Some("collie")), (273, 0.015174894, Some("dingo, warrigal, warragal, Canis dingo")), (248, 0.014367299, Some("Eskimo dog, husky"))] } }
1: Y { Probs:  { Top5: [(284, 0.9907692, Some("siamese cat, Siamese")), (285, 0.0015794479, Some("Egyptian cat")), (174, 0.0015189401, Some("Norwegian elkhound, elkhound")), (225, 0.00031838714, Some("malinois")), (17, 0.00027021166, Some("jay"))] } }
2: Y { Probs:  { Top5: [(387, 0.94238573, Some("lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens")), (368, 0.0029994072, Some("gibbon, Hylobates lar")), (277, 0.0016564301, Some("red fox, Vulpes vulpes")), (356, 0.0015081967, Some("weasel")), (295, 0.001427932, Some("American black bear, black bear, Ursus americanus, Euarctos americanus"))] } }

```
