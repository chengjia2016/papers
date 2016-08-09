# Deep Visual-Semantic Alignments for Generating Image Descriptions

## 摘要

我们呈现了一个可以用自然语言描述图片的模型。我们的方法通过从图片和对应描述的数据集中学习他们之间的联系。我们的联合模型是把处理图片的CNN和用于处理句子的双向RNN(bidirectional RNN)组合起来，通过多模型embedding组合得到一个结构化的目标。本作然后描述如何使用推理出的联系在多模型RNN架构中去学习如何生成图片的描述。结果表明本作的联合模型在数据集（Flickr8K, Flickr30K, MSCOCO）检索实验中表现优异。最后，我们证明了通过使用生成的图片描述，极大的提高了在全图和一个新的region-level标注数据集的检索基线。

## 代码

<https://github.com/karpathy/neuraltalk2>

Torch

第一版代码

<https://github.com/karpathy/neuraltalk>

python