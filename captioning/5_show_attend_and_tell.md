# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

## 2016

## 一句话摘要

有Tensorflow代码, 引入attention。

## 摘要

受到近期机器翻译和物体检测的启发，本作展示一个基于注意力(attention)的模型用于自动描述图片。本文描述如何使用标准的backpropagation技术和概率上的最大化variational lower bound来以一种确定的方式训练模型。同时通过可视化来展示模型是如何通过观察一个静态的图片来得到相应的描述。在3个数据集上验证本作: Filckr8k, Filckr30k, MSCOCO。

Inspired by recent work in machine translation and object detection, we introduce an attention based model that automatically learns to describe the content of images. We describe how we can train this model in a deterministic manner using standard backpropagation techniques and stochastically by maximizing a variational lower bound. We also show through visualization how the model is able to automatically learn to fix its gaze on salient objects while generating the corresponding words in the output sequence. We validate the use of attention with state-of-the-art performance on three benchmark datasets: Flickr8k, Flickr30k and MS COCO.

## 代码

<https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow>

Tensorflow

## 作者

Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, Yoshua Bengio
