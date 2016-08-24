# 用python和theano实现GRU和LSTM

上一篇介绍了RNN,然后实现了RNNNumpy和RNNTheano两个版本。RNNNumpy是手写前向和后项过程。而RNNTheano只需要手写出前向过程, 后向过程留给Theano去处理。本篇我们来继续LSTM(Long Short Term Memory)和GRU(Gated Recurrent Unit)。LSTM是1997年出现的，GRU出现在2014年作为LSTM的一个变体。

<!-- more -->

## LSTM 网络
