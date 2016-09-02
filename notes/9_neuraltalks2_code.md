# 详解neuraltalks2代码

我们已经把neuraltalks2的环境搭建好了, 并制作了docker image方便后续部署。模型经过了初步训练, 现在进入精调阶段, 当前的CIDEr值是最高达到0.853。  
在更早之前我们也学习过了RNN, 并有过一段带啊实现过char粒度的语言模型。  
neuraltalks2是一个RNN+CNN的模型, RNN负责语言模型部分, CNN负责图片特征提取部分, 在这些基础上，现在让我们一起深入的看下neuraltalks2的代码。

##
