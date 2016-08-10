# Deep Visual-Semantic Alignments for Generating Image Descriptions

## 2015

## 一句话摘要

本作即neuraltalk2的文章, Torch实现。标准的image captioning。

## 摘要

我们呈现了一个可以用自然语言描述图片的模型。我们的方法通过从图片和对应描述的数据集中学习他们之间的联系。我们的联合模型是把处理图片的CNN和用于处理句子的双向RNN(bidirectional RNN)组合起来，通过多模型embedding组合得到一个结构化的目标。本作然后描述如何使用推理出的联系在多模型RNN架构中去学习如何生成图片的描述。结果表明本作的联合模型在数据集（Flickr8K, Flickr30K, MSCOCO）检索实验中表现优异。最后，我们证明了通过使用生成的图片描述，极大的提高了在全图和一个新的region-level标注数据集的检索基线。

We present a model that generates natural language descriptions of images and their regions. Our approach lever- ages datasets of images and their sentence descriptions to learn about the inter-modal correspondences between language and visual data. Our alignment model is based on a novel combination of Convolutional Neural Networks over image regions, bidirectional Recurrent Neural Networks over sentences, and a structured objective that aligns the two modalities through a multimodal embedding. We then describe a Multimodal Recurrent Neural Network architecture that uses the inferred alignments to learn to generate novel descriptions of image regions. We demonstrate that our alignment model produces state of the art results in retrieval experiments on Flickr8K, Flickr30K and MSCOCO datasets. We then show that the generated descriptions significantly outperform retrieval baselines on both full images and on a new dataset of region-level annotations.

## 代码

<https://github.com/karpathy/neuraltalk2>

Torch

第一版代码

<https://github.com/karpathy/neuraltalk>

python


## 作者

Andrej Karpathy, Li Fei-Fei
