# CascadeDetector

PyTorch's Cascade Detector Framework

## Requirement

- Python 3
- PyTorch >= 1.0.0
- Numpy
- OpenCV
- Pillow

## Features

- [MTCNN](https://arxiv.org/abs/1604.02878) Framework
- Customize P, R and O Net, You can plugin your own implemented P, R and O Net
- Highly Customized Config
- Multiple Backend Support (OpenCV and Pillow)
- Multiple Device Support (CPU and GPU)

## Example

![result](assets/example_result.jpg)

## Quick Start

```python
from PIL import Image
image = Image.open("example.jpg")
net = CascadeDetector(your_pnet, your_rnet, your_onet)
bboxes, landmarks = net(image)
```

Usage is shown in [notebook](notebooks/example.ipynb)

## Reference

[1]. Zhang, Kaipeng, Zhanpeng Zhang, Zhifeng Li and Yu Qiao. “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.” IEEE Signal Processing Letters 23 (2016): 1499-1503.