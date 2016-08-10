# DenseCap: Fully Convolutional Localization Networks for Dense Captioning

## 2016

## 一句话摘要

有Torch代码, 本作是neuraltalk2的团队后续作品，特点是生成dense and rich的标注，即想尽的把图片中的物品描述出来，且只需要1次forward pass。

## 摘要

本作引入了稠密描述任务(dense captioning task), 即让计算机视觉系统同时定位并描述图片中的显著区域(salient regions)。稠密描述任务当描述只由1个词组成时就一般化为物体检测(object detection)，当检测区域覆盖整个图片时就是图片描述(Image Captioning)。为了将定位和描述任务合在一起，我们提出了全连通卷积定位网络(FCLN, Fully Convolutional Localization Network)架构，用于使用一个单一并高效的前向过程(不需要额外的regions proposals)处理图片，并可以端到端的通过一轮优化进行训练。本架构由一个CNN，以及新的稠密定位层，和RNN语言模型（用于生成句子）构成。我们在Visual Genome数据集(包含94,000图片和4,100,000基于区域的描述)上评测，观察到无论是速度上还是准确性都超过当前其他近期方法。

We introduce the dense captioning task, which requires a computer vision system to both localize and describe salient regions in images in natural language. The dense captioning task generalizes object detection when the descriptions consist of a single word, and Image Captioning when one predicted region covers the full image. To address the localization and description task jointly we propose a Fully Convolutional Localization Network (FCLN) architecture that processes an image with a single, efficient forward pass, requires no external regions proposals, and can be trained end-to-end with a single round of optimization. The architecture is composed of a Convolutional Network, a novel dense localization layer, and Recurrent Neural Network language model that generates the label sequences. We evaluate our network on the Visual Genome dataset, which comprises 94,000 images and 4,100,000 region-grounded captions. We observe both speed and accuracy improvements over baselines based on current state of the art approaches in both generation and retrieval settings.


## 代码

<https://github.com/jcjohnson/densecap>

Torch

## 作者

Justin Johnson, Andrej Karpathy, Li Fei-Fei
