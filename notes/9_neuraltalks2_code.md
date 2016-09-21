# 详解neuraltalks2代码

我们已经把neuraltalks2的环境搭建好了, 并制作了docker image方便后续部署。模型经过了初步训练, 现在进入精调阶段, 当前的CIDEr值是最高达到0.853。  
在更早之前我们也学习过了RNN, 并有过一段带啊实现过char粒度的语言模型。  
neuraltalks2是一个RNN+CNN的模型, RNN负责语言模型部分, CNN负责图片特征提取部分, 在这些基础上，现在让我们一起深入的看下neuraltalks2的代码。

## 主体流程

先跳过validation过程, validation过程是每隔`opt.save_checkpoint_every`次进行一次，一般是2500轮迭代。

### Loss Function

先设置cnn和lm的网络状态到train
```
protos.cnn:training()
protos.lm:training()
```

给grad_params清零
```
grad_params.zero()
```

option: 非CNN精调时跳过清零 `cnn_grad_params`

#### forward pass

先用`loader:getBatch`获取一批训练数据，包括image向量和label向量。  

然后调用`net_utils.prepro`对图像进行预处理: 把图片由256*256转换成224*224的区域。把imgs对应的向量转换到GPU(如果用了gpu)，然后把图像用vgg_mean"蒙板"处理下, 直观理解就是把所有的图片的(224,224)区域的RGB色域都减去vgg_mean中的RGB值.

```
if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
    print("net_utils.prepro, init net_utils.vgg_mean", net_utils.vgg_mean)
end
net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match
print("net_utils.prepro, after typsAs(imgs) net_utils.vgg_mean", net_utils.vgg_mean)
imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))
```

这个vgg_mean的颜色如下:
![](9_vgg_mean.png)

将预处理后的data.images喂给CNN网络`protos.cnn:foward(data.images)`  

这个CNN网络如下:

```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) ->
  (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) ->
  (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> (24) ->
  (25) -> (26) -> (27) -> (28) -> (29) -> (30) -> (31) -> (32) -> (33) -> (34) -> (35) -> (36) -> (37) -> (38) -> (39) -> (40) -> output]
  (1): cudnn.SpatialConvolution(3 -> 64, 3x3, 1,1, 1,1)
  (2): cudnn.ReLU
  (3): cudnn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
  (4): cudnn.ReLU
  (5): cudnn.SpatialMaxPooling(2x2, 2,2)
  (6): cudnn.SpatialConvolution(64 -> 128, 3x3, 1,1, 1,1)
  (7): cudnn.ReLU
  (8): cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1)
  (9): cudnn.ReLU
  (10): cudnn.SpatialMaxPooling(2x2, 2,2)
  (11): cudnn.SpatialConvolution(128 -> 256, 3x3, 1,1, 1,1)
  (12): cudnn.ReLU
  (13): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
  (14): cudnn.ReLU
  (15): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
  (16): cudnn.ReLU
  (17): cudnn.SpatialMaxPooling(2x2, 2,2)
  (18): cudnn.SpatialConvolution(256 -> 512, 3x3, 1,1, 1,1)
  (19): cudnn.ReLU
  (20): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (21): cudnn.ReLU
  (22): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (23): cudnn.ReLU
  (24): cudnn.SpatialMaxPooling(2x2, 2,2)
  (25): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (26): cudnn.ReLU
  (27): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (28): cudnn.ReLU
  (29): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (30): cudnn.ReLU
  (31): cudnn.SpatialMaxPooling(2x2, 2,2)
  (32): nn.View(-1)
  (33): nn.Linear(25088 -> 4096)
  (34): cudnn.ReLU
  (35): nn.Dropout(0.500000)
  (36): nn.Linear(4096 -> 4096)
  (37): cudnn.ReLU
  (38): nn.Dropout(0.500000)
  (39): nn.Linear(4096 -> 512)
  (40): cudnn.ReLU
}
```

原始的VGG16 CNN是38层，最后两层是这里加入的  

这里先不深入这个CNN，把其当作一个黑盒处理，forward输入一批图片, 得到这批图片的一组图像特征feats(16,512)  

将feats喂给`protos.expander:foward(feats)`  
feats是(16,512)的Tensor, 表示16个图片的图像特征, 经过expander层后, 变成(16*5,512)的Tensor, 把原来的每个图片对应的512图像特征复制5份  
train.lua中调用的是`expander:forward(feats)`, 按照Torch.nn的文档, 所有继承了nn.Module的层, 不建议覆盖forward(), 应该实现updateOutput()

将经过expander层处理后的expanded_feats (16*,512)的向量, 联同data.labels喂给LM层  
`local logprobs = protos.lm:forward{expanded_feats, data.labels}`  

##### LM

lm:forward在LanguageModel.lua中nn.LanguageModel的updateOutput()  
参数是外层通过{}传进来的,是个table,是两个参数imgs和seqs  
先确认self.clones存在  
**目前还不理解为何要self.clones**

###### LM的createClones()
会创建self.clones和self.lookup_tables  
都是table, 都含有16+2的长度, 16是每个caption的长度, 2表示开头和结尾

### option: 调整LM和CNN的学习率

默认不调整

### LM参数更新

```
adam(params,
     grad_params,
     learning_rate,
     opt.optim_alpha,  -- 0.8
     opt.optim_beta,   -- 0.999
     opt.optim_epsilon,-- 1e-8
     optim_state)
```

### CNN参数更新

只有在opt.finetune_cnn_after>0时才会更新CNN的参数  
finetune指的就是更新CNN网络
训练的第一阶段只更新LM的参数  
第二阶段进入opt.finetune_cnn_after 0表示从第一阶段的checkpoint开始训练，
