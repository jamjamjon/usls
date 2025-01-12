# LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation

Pixel-wise semantic segmentation for visual scene understanding not only needs to be accurate, but also efficient in order to find any use in real-time application. Existing algorithms even though are accurate but they do not focus on utilizing the parameters of neural network efficiently. As a result they are huge in terms of parameters and number of operations; hence slow too. In this paper, we propose a novel deep neural network architecture which allows it to learn without any significant increase in number of parameters. Our network uses only 11.5 million parameters and 21.2 GFLOPs for processing an image of resolution 3x640x360. It gives state-of-the-art performance on CamVid and comparable results on Cityscapes dataset. We also compare our networks processing time on NVIDIA GPU and embedded system device with existing state-of-the-art architectures for different image resolutions.

## Official Repository

The official paper can be found on: [LinkNet](https://arxiv.org/abs/1707.03718)

## Example

Refer to the [example](../../../examples/linknet)
