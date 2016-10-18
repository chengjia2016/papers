# 针对广告图片的caption

计划利用py faster rcnn的输出结果作为caption的pre-trained模型  
然后整理一批广告图片的caption训练数据  
重新训练针对广告图片的caption模型

把广告图片和对应的label整理出来, 加入到模型中, 让即便是没有图片-caption的训练数据也能认识物体。参考: <https://people.eecs.berkeley.edu/~lisa_anne/dcc_project_page.html>

## 中心思想
我们对captioning的首要诉求是利用其输出层或者中间层的结果计算 **用户-素材的语义相关性** ，其次才是captioning直接服务于素材理解。
同时运用图片和文字数据，应该比单独用图片或者文字有更多的输入信息量。
这个之前贴过。captioning和图片-文字跨模态的检索有共同的技术基础。

最终想要达到的例子, 比如我询问"京东最近红色的女装", 那么可以把众多素材中符合的挑选出来。用户的状态可以从过去n天的日志中抽取，分析过去n天的所见素材，分别是什么东西，这些东西的集合就是用户的喜好。用这个用户喜好跟订单素材算距离。

caption, cross-modal retrival, visual-semantic-embedding这些都是工具, 用这些工具去完成上面具体的目标。


## 思考
训练数据是一组用户侧的特征，加上定单侧的特征, 计算出用户对各个订单的值, 值最高的就是这个用户最想要点的广告。

原来对于订单的素材，只是一个素材id, 这个粒度太细了。如果能把素材的图像和素材的文字变成一个特征加入到训练数据中，应该会有信息量。

对于一个图片，通过caption模型，输出一组用于描述其的文字，而文字是来自于不断的从模型中采样得到

## Try List

1 Convolution-DSSM
2 Deep CTR paper

## ref:
<https://github.com/ryankiros/visual-semantic-embedding>
