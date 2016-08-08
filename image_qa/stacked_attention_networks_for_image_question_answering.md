# Stacked Attention Networks for Image Question Answering

## 摘要
本作展示层叠注意力网络(stacked attention networks, SANs)用于学习如何以一幅图片为基础的自然语言问题。SANs用问题的语意表述，在图片中搜索与问题有关的区域。我们认为图片问答通常需要多步推理。因此，我们设计了一个多层SAN用于一步步的推理答案。实验在4个图像问答数据集上表明此方法优于其他。通过注意力层的可视化，我们发现SAN可以逐层的愈加锁定答案相关的区域。

This paper presents stacked attention networks (SANs) that learn to answer natural language questions from im- ages. SANs use semantic representation of a question as query to search for the regions in an image that are related to the answer. We argue that image question answering (QA) often requires multiple steps of reasoning. Thus, we develop a multiple-layer SAN in which we query an image multiple times to infer the answer progressively. Experi- ments conducted on four image QA data sets demonstrate that the proposed SANs significantly outperform previous state-of-the-art approaches. The visualization of the atten- tion layers illustrates the progress that the SAN locates the relevant visual clues that lead to the answer of the question layer-by-layer.

## 代码
https://github.com/ zcyang/imageqa-san

Theano