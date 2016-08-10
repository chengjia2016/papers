# Show and Tell: A Neural Image Caption Generator

## 2015

## 一句话摘要

Tensorflow代码

## 摘要

自动描述图片是人工智能中图像领域的基础问题，连接了计算机视觉(CV)和自然语言处理(NLP)。本作呈现了一个基于深度递归架构的生成模型，将近期在计算机视觉和机器翻译中的成果合并，来生成可以描述图片的句子。模型通过最大化目标图片对应句子的likelihood进行训练。在多个数据集上的实验表明，模型的准确度和语言的流畅度单纯的受图片描述语言的影响。从定性和定量来看本作模型颇为准确。比如当前其他近期工作在Pascal数据集上的BLEU-1得分是25（越高越好），本作是59，人类表现是69。在Flickr30K上得分从56提升到66，在SBU数据机上得分从19到28。最后在COCO数据集上，BLEU-4得分是27.7，是最近的最高分。

Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. In this paper, we present a generative model based on a deep recurrent architecture that combines recent advances in computer vision and machine translation and that can be used to generate natural sentences describing an image. The model is trained to maximize the likelihood of the target description sentence given the training image. Experiments on several datasets show the accuracy of the model and the fluency of the language it learns solely from image descriptions. Our model is often quite accurate, which we verify both qualitatively and quantitatively. For instance, while the current state-of-the-art BLEU-1 score (the higher the better) on the Pascal dataset is 25, our approach yields 59, to be compared to human performance around 69. We also show BLEU-1 score improvements on Flickr30k, from 56 to 66, and on SBU, from 19 to 28. Lastly, on the newly released COCO dataset, we achieve a BLEU-4 of 27.7, which is the current state-of-the-art.

# 代码

<https://github.com/jazzsaxmafia/show_and_tell.tensorflow>

Tensorflow

# 作者

Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
