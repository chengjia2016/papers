# Generation and Comprehension of Unambiguous Object Descriptions

## CVPR 2016

## 一句话摘要

尽可能无奇异的描述图片中某个物体。在图中标记一个点，重点描述这个点，不同于COCO数据集是描述整个图片。

## 摘要

本作提出了一种可以生成图片中一个特定物体或者区域的无歧义描述(或者叫referring expression，我理解就是特定区域表述)，即可以通过理解这个表述推断出描述的是图片里的哪个物体。本作通过将图片中其他可能引起歧义的潜在物体去除可以得到更好的对特定区域或者物体的描述。本作模型收到近期的image captioning方法启发，尽管image captioning难于评测，但是本作的目标可以得到客观的评估。本作还展示了一个用于referring expression的数据集（基于MS-COCO）。我们在github上发布了这个数据集以及工具包用于可视化和评估: https://github.com/mjhucla/Google_Refexp_toolbox。

We propose a method that can generate an unambiguous description (known as a referring expression) of a specific object or region in an image, and which can also comprehend or interpret such an expression to infer which object is being described. We show that our method outperforms previous methods that generate descriptions of objects without taking into account other potentially ambiguous objects in the scene. Our model is inspired by recent successes of deep learning methods for image captioning, but while image captioning is difficult to evaluate, our task allows for easy objective evaluation. We also present a new large-scale dataset for referring expressions, based on MS- COCO. We have released the dataset and a toolbox for visualization and evaluation, see https://github.com/ mjhucla/Google_Refexp_toolbox.

## 代码

<https://github.com/mjhucla/Google_Refexp_toolbox>
是toolbox的代码

## 作者

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, Kevin Murphy
