## Quick Start

```shell
cargo run -r -F cuda --example florence2 -- --device cuda --dtype fp16
```


```Shell
Task: Caption(0)
Ys([Y { Texts: [Text("A green car parked in front of a yellow building.")] }, Y { Texts: [Text("A group of people walking down a street next to a bus.")] }])

Task: Caption(1)
Ys([Y { Texts: [Text("The image shows a green car parked in front of a yellow building with two brown doors. The car is on the road, and the building has a wall and a tree in the background.")] }, Y { Texts: [Text("The image shows a group of people walking down a street next to a bus, with a building in the background. The bus is likely part of the World Electric Emission Bus, which is a new bus that will be launched in Madrid. The people are walking on the road, and there are trees and a sign board to the left of the bus.")] }])

Task: Caption(2)
Ys([Y { Texts: [Text("The image shows a vintage Volkswagen Beetle car parked on a cobblestone street in front of a yellow building with two wooden doors. The car is a light blue color with silver rims and appears to be in good condition. The building has a sloping roof and is painted in a bright yellow color. The sky is blue and there are trees in the background. The overall mood of the image is peaceful and serene.")] }, Y { Texts: [Text("The image shows a blue and white bus with the logo of the Brazilian football club, Cero Emisiones, on the side. The bus is parked on a street with a building in the background. There are several people walking on the sidewalk in front of the bus, some of them are carrying bags and one person is holding a camera. The sky is blue and there are trees and a traffic light visible in the top right corner of the image. The image appears to be taken during the day.")] }])
```

## Results

| Task   |  Demo |
| -----| ------|
|Caption-To-Phrase-Grounding | <img src='https://github.com/jamjamjon/assets/releases/download/florence2/Caption-To-Phrase-Grounding-car.png' alt=''> |
| Ocr-With-Region | <img src='https://github.com/jamjamjon/assets/releases/download/florence2/Ocr-With-Region.png' alt=''>|
|  Dense-Region-Caption | <img src='https://github.com/jamjamjon/assets/releases/download/florence2/Dense-Region-Caption-car.png' alt=''>|
| Object-Detection | <img src='https://github.com/jamjamjon/assets/releases/download/florence2/Object-Detection-car.png' alt=''>|
| Region-Proposal | <img src='https://github.com/jamjamjon/assets/releases/download/florence2/Region-Proposal.png' alt=''>|
| Referring-Expression-Segmentation | <img src='https://github.com/jamjamjon/assets/releases/download/florence2/Referring-Expression-Segmentation.png' alt=''>|


