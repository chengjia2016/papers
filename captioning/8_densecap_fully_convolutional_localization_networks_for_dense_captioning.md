# DenseCap: Fully Convolutional Localization Networks for Dense Captioning

## 摘要

本作引入了稠密描述任务(dense captioning task), 即让计算机视觉系统同时定位并描述图片中的显著区域(salient regions)。稠密描述任务当描述只由1个词组成时就一般化为物体检测(object dectection)，当检测区域覆盖整个图片时就是图片描述(Image Captioning)。为了将定位和描述任务合在一起，我们提出了全连通卷积定位网络(FCLN, Fully Convolutional Localization Network)架构，用于使用一个单一并高效的前向过程(不需要额外的regions proposals)处理图片，并可以端到端的通过一轮优化进行训练。本架构由一个CNN，以及新的稠密定位层，和RNN语言模型（用于生成句子）构成。我们在Visual Genome数据集(包含94,000图片和4,100,000基于区域的描述)上评测，观察到无论是速度上还是准确性都超过当前其他近期方法。

## 代码

<https://github.com/jcjohnson/densecap>

Torch

## 作者

Justin Johnson, Andrej Karpathy, Li Fei-Fei
