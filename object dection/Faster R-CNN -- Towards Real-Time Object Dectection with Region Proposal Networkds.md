# Faster R-CNN: Towards Real-Time Object Dectection with Region Proposal Networkds

## 摘要
最新的物体检测网络是基于region proposal算法去猜测物体位置的。更进一步的算法比如SPPnet 和 Fast R-CNN 减少检测网络的训练时间, 揭示region proposal是计算瓶颈。本作中, 我们引入了Region Proposal Network(RPN), 让检测网络共享full-image convolutional特征，这样极大的削减了region proposals的计算成本。RPN是一个fully convolutional网络，同时预测物体边界和每个位置上物体得分。通过端到端的训练得到RPN, 生成高质量的region proposal, 在Fast R-CNN中用于检测。本作进一步的将RPN和Fast R-CNN合并到一个单一网络中, 共享他们的卷积特征，通过attention机制（一个近期神经网络流行术语），让RPN组建告诉网络该去检测图片中的哪个位置。对于VGG16模型，本作检测系统可以在GPU上达到5fps的帧率(包括所有步骤), 另外在PASCAL VOC 2007, 2012 和 MS COCO数据集上，使用每个图片300 proposal的情况下达到很好的物体检测准确度。Faster R-CNN 和 RPN 是近期多个比赛中夺得好成绩的重要基础。代码已经公开。

## 1 介绍
近期在物体检测的进展是在region proposal方法和regin-based卷积神经网络的成功下取得的。原来的region-based CNN计算成本很高, 但通过在proposal间共享卷积大幅的减少计算成本。到了Fast R-CNN, 通过使用极深网络可以达到接近实时(在忽略掉region proposal话费的时间下)。目前, proposal是(test-time)计算的瓶颈所在。

Region proposal方法依赖于inexpensive features和economical inference schemes(我理解是简单特征和简单的推理方法，简单是计算简单)。`Selective Search` 是一个最流行的方法，贪心的合并基于engineered低级特征超像素(superpixels)。当与高效的检测网络相比，Selective Search大约是落后1个数量级的缓慢，大约在CPU下是2秒处理一幅图片。 `EdgeBoxes` 目前在proposal质量和速度上取得了一个较好的权衡，大约0.2秒处理一幅图片。但是，region proposal仍然在检测网络中消耗很多的时间。

有人指出快速的region-based CNN可以利用到GPU，但是region proposal方法是在CPU下实现的, 这样比较这二者的运行效率是不公平的。一个简单的方法来加速proposal计算是在GPU下重新实现。这个也许是个有用的工程解决方案，但是重新实现忽略了down-stream检测网络, 因此失去了共享计算的机会。

本作中给出了个算法上的更改：通过深度卷积网络计算proposal，是一种可以优雅的解决方案，让proposal的计算几乎在检测网络计算外不消耗其他计算资源。我们引入了`Region Proposal Networks`, 与检测网络共享卷积层。通过在test-time时共享卷积，计算proposal的时间会非常小（约10ms每个图片）。

我们观察到region-based检测器(比如Fast R-CNN)用到的卷积特征，可以用于生成region proposal。在这些卷积特征之上, 我们构建一个RPN：增加一些额外的卷积层用于同时回归region bounds和各个规律网格位置上物体预测的分数(objectness scores)。RPN是一个fully convolutional network(FCN，全连通卷积网络)，可以端到端的训练，用于生成检测proposals。

RPN的设计目标是可以高效的预测region proposal，可以缩放和旋转图像。相比于其他流行的使用pyramids（金字塔）of images的方法或者pyramids of filters的方法（我理解就是各种表示多个缩放和大小的方法），本作引入了`anchor boxes`，给不同尺寸缩放用references来表示，可以避免枚举images或者filters的多种缩放和方向。这种model在使用单一尺寸的图片进行训练和测试时在计算速度上表现很好。

为了将RPN和Fast R-CNN物体检测网络合并到一起，我们提出了一种训练机制：在调优region proposal任务和调优物体检测之间进行交替，同时保持proposals是固定的。这种训练机制可以快速收敛，并可以得到一个单一的网络，让两个任务共享卷积特征。

我们在PASCAL VOC物体检测基准上评估了RPN+Fast R-CNN的方法，可以得到比Selective Search+Fast R-CNN更好的准确率。同时，我们的方法几乎避免了Selective Search中在test-time期间的所有计算负担（在proposals计算时间每幅图片只有10ms）。通过使用极深网络，我们的检测方法仍然可以在GPU上达到**5fps**的帧率。同时，我们还在MS COCO数据集上进行了测试。有python和matlab的公开代码：

Matlab: <https://github.com/shaoqingren/faster_rcnn>

Python: <https://github.com/rbgirshick/py-faster-rcnn>

本作的初稿发布后，使用RPN的框架和Faster R-CNN被广泛采纳，并衍生出了其他的方法，比如`3D物体检测`，`part-based 物体检测`, `instance segmentation`, `image captioning`。并在商业系统中应用，比如`Pinterets`。

在ILSVRC和COCO 2015竞赛里，Faster R-CNN和RPN是众多一等模型的重要基础，在多个领域都有建树: ImageNet detection, ImageNet localization, COCO detection, COCO segmentation。 RPNs 完全是从数据中学习如何去propose regions， 可以从更深的网络和更具表达性的特征（比如101层的residual net）中获得收益。


## 2 相关工作

## 3 Faster R-CNN

### 3.1 Region Proposal Networks

#### 3.1.1 Anchors

#### 3.1.2 Loss Function

#### 3.1.3 Training RPNs

### 3.2 RPN和Fast R-CNN共享特征

### 3.3 实现细节

## 4 实现

### 4.1 PASCAL VOC 实验

### 4.2 MS COCO 实验

### 4.3 从MS COCO到PASCAL VOC

## 5 结论



