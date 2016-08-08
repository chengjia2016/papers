#Deep Compositional Captioning: Describing Novel Object Categories Without Paired Training Data

##摘要

目前最新的深度神经网络模型在image captioning领域取得了突飞猛进，他们极大的依赖图片与句子的组成的成对语料库。本作提出了Deep Compositional Captioner(DCC，深度合成解说器)，用于生成新物体（没有出现在成对儿的图片-句子数据集中）的描述。我们的方法是利用了物体识别数据库和外部文字语料库以及语意相似概念上的迁移知识。当前的深度解说器模型只能描述那些在成对儿的图片-文字语料库里包括的物体，尽管他们使用了预先训练好的物体识别模型（比如使用ImageNet）。相反，本作模型可以描述新物体以及与其他物体之间的联系。我们在MSCOCO数据集上证明了描述新物体的能力，以及在ImageNet的图片中描述没有在训练集（图片-文字对）中出现过的物体。更进一步的，我们扩展到了描述视频片段上，结果证明DCC在生成新物体上面比现有的其他图片或视频解说器更好。

While recent deep neural network models have achieved promising results on the image captioning task, they rely largely on the availability of corpora with paired im- age and sentence captions to describe objects in context. In this work, we propose the Deep Compositional Cap- tioner (DCC) to address the task of generating descriptions of novel objects which are not present in paired image- sentence datasets. Our method achieves this by leveraging large object recognition datasets and external text corpora and by transferring knowledge between semantically sim- ilar concepts. Current deep caption models can only de- scribe objects contained in paired image-sentence corpora, despite the fact that they are pre-trained with large object recognition datasets, namely ImageNet. In contrast, our model can compose sentences that describe novel objects and their interactions with other objects. We demonstrate our model’s ability to describe novel concepts by empir- ically evaluating its performance on MSCOCO and show qualitative results on ImageNet images of objects for which no paired image-sentence data exist. Further, we extend our approach to generate descriptions of objects in video clips. Our results show that DCC has distinct advantages over existing image and video captioning approaches for generating descriptions of new objects in context.

## 备注

输入不光有成对出现的图片+文字，还有图片本身的描述，和名词本身的描述，与成对出现的数据一起进行训练。