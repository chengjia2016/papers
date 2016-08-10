#Sequence to Sequence -- Video to Text

## 摘要

现实世界的视频通常复杂多样，用于生成开放领域视频描述的方法应该对于时间结构敏感，并且可以允许可变长的输入（一系列帧）和输出（一系列词语）。为了打成这个目标，我们提出了端到端序列到序列的模型用于生成视频的描述。利用LSTM单元的RNN。使用成对出现的视频-句子用来训练，将一系列视频帧与一系列词语连接起来用于生成一段视频剪辑的描述。模型可以学到帧序列与一系列词语对应的时间结构，即语言模型。我们使用多个模型的变种在youtube和两个电影描述数据库（M-VAD和MPII-MD）上发现了多个视觉特征。

## 代码
<https://github.com/jazzsaxmafia/video_to_sequence>

Tensorflow

[demo](https://vsubhashini.github.io/s2vt.html)

## 作者

Subhashini Venugopalan
