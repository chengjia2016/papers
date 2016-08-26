# 不可思议的RNN(有代码详解)

现在我已经对RNN有了一些了，本文从RNN的特点入手，紧接着用100行的代码实现了一个简单的RNN模型，且配有详细的代码解释。

## 理解RNN

### 性质：序列

RNN的重点是输入和输出都可以是序列，而不是一个固定长度。这是其他网络结构不具备的能力。

5个学习类型:
* one to one: image classification
* one to many: image captioning
* many to one: sentiment analysis
* many to many I: machine translation
* many to many II: video classification

<!-- more -->

![](images/4_5_types_network.jpeg)

### 即便数据不是序列的，可以“序列的训练”

序列的输入或者输出可能比较稀少，但是可以把固定的输入输出按照“序列”的形式处理。

### RNN的计算

一个最简单的RNN

```python
rnn = RNN()
# x is an input vector, y is the RNN's output vector
y = rnn.step(x)
```

RNN类里面维护了一个内部隐向量(隐状态)$h$，用输入$x$去更新这个隐状态。

比如这个原始RNN前向过程: 输出基于当前输入和历史状态（隐状态h）

```python
def step(self, x):
  self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
  y = np.dot(self.W_hy, self.h)
```

前向更新公式:  

$h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t)$  
$y = W_{hy}h$  

$tanh$激活函数把值限制在[-1,1]之间  
RNN有三个参数$W_{hh}, W_{xh}, W_{hy}$  
`np.dot`是矩阵乘法  

跟`2_origin_rnn.md`中的前向公式对比下, 基本一致。  

$s_t = tanh(Ux_t+Ws_{t-1})$  
$o_t = softmax(Vs_t)$  

也是三个参数，对应关系如下:  

$W_{xh} - U$ 是把输入 $x$ 转换成 $h$ 空间的矩阵  
$W_{hh} - W$ 是把前一时刻的隐状态 $h$ 转到到当前状态的矩阵  
$W_{hy} - V$ 是从隐状态空间转换到输出空间的矩阵  

### 加深一点

给RNN加上“深”的帽子，多摞几层即可。

```python
y1 = rnn1.step(x)
y = rnn2.step(y1)
```

即我们有两个RNN。

### 更酷一点

实际中不使用上面的原始RNN，而是用LSTM。因为其不同的更新公式（更新公式不同，则其反向传导公式也不同）

## 字节级别的语言模型

![](images/4_charseq.jpeg)

上图是隐层具有3个神经元的RNN实例。输入是字节"hell", 展示的是前向过程，输出的字典包含4种字母(h,e,l,o)。上面的计算过程是:
'h' -> 'e'
'e' -> 'i'
'l' -> 'l'
'l' -> 'o'
每一个字母输入都有一个输出，预测这个字母的下一个字母是什么，而隐状态持续更新。

训练数据中，每个输入都有真实对应的输出，根据真实的输出和模型预测的输出，朝着对的方向调整参数（通过反向传播算法根据梯度更新参数）。按照这样的参数更新方法，当我们使用相同的训练数据输入到模型时，会发现预测正确的字母的值（比如第一次见到 'h' 时输出是正确值 'e' 是2.2）变高了（变成了2.3）。不断这这样重复, 直到模型的预测值跟训练数据的实际输出值一样，模型就收敛了。

值得注意的是第一次遇见'l'时输出的是'l', 第二次遇见'l'是输出的是'o'。这就体现了RNN不仅依赖于当前输入，还需要内部维护的状态来追踪上下文的强大特性。


### numpy版rnn代码: min-char-rnn.py

多说无意，看代码加深理解，[100行的numpy版代码](
https://gist.github.com/wang-yang/31e673e4792954305a311fc2356de452)，在原版基础上添加了一些注释, 并有代码详解。


运行方法: `python min-char-rnn.py`

#### **代码流程如下:**

1. 首先读入输入文件: data
2. 获取字符集: chars
3. 初始化字符到索引的映射和索引到字符的映射: char_to_ix, ix_to_char
4. 设置隐层的神经元个数 hidden_size
5. RNN需要记忆的步数(即RNN展开大小) seq_length
6. 设置学习率 learning_rate
7. 初始化模型参数:
   1. Wxh, 输入到隐层的转化矩阵
   2. Whh, 旧隐层到新隐层的转化矩阵
   3. Why, 隐层到输出的转化矩阵
   4. bh, 隐层的bias
   5. by， 输出层的bias
8. 初始化n和p, n用来记录while跑了多少轮, 每100轮输出当前模型生成的样本和训练进度
9. 初始化mWxh, mWhh, mWhy, mbh, mby, 这是adagrad更新公式需要的中间变量
10. 初始化smooth_loss
11. 进入while循环, 我们不断的学习data, 以步长seq_length不断的重复
12. 准备这次训练的训练数据: inputs和targets
13. 每当100轮打印一次模型生成的文本
14. 调用lossFun进行前向和后向的计算，输入是inputs, targets, hprev。hprev是当前的隐状态(如果神经元个数是100，隐状态向量就是(100,1)的样子)。输出是loss, dWxh, dWhh, dWhy, dbh, dby, hprev。
15. 更新loss
16. 利用dWxh, dWhh, dWhy, dbh, dby更新Wxh, Whh, Why, bh, by。使用adagrad方法
17. 更新n和p

#### **lossFun**

1. 初始化xs, hs, ys, ps
2. 前向过程, 从0开始到seq_length次的展开

先看完整代码  

```python
xs[t] = np.zeros((vocab_size,1))
xs[t][inputs[t]] = 1
hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
ys[t] = np.dot(Why, hs[t]) + by
ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
loss += -np.log(ps[t][targets[t],0])
```

利用one-hot的形式(1-of-k形式)初始化进入网络的输入:

```python
xs[t] = np.zeros((vocab_size,1))
xs[t][inputs[t]] = 1
```

这两个公式的实现:  
计算隐状态: $h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t)$  
**unnormalized** log probabilities for next char: $y = W_{hy}h$  

```python
hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
ys[t] = np.dot(Why, hs[t]) + by
```

使用 **softmax** , 得到 **normalized** probabilities for next chars
```python
ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
```

**注**: **softmax** 又叫 **normalized exponential** ，把$K$维实数向量$z$，投影到$K$维向量$\sigma(z)$, 但取值范围变成(0,1), 且所有维度的和是1。

定义: $\sigma(z)_j=\frac{e^{z_{j}}}{\sum_{k=1}^{K}e^{z_{k}}}$

最后更新loss, **这里不理解为何这样设置loss**

```python
loss += -np.log(ps[t][targets[t],0])
```
**注**: `ps[t][targets[t], 0]`,比如`targets[t]=3`就是`ps[t][3,0]]`，numpy中的array支持这种形式的索引, 取的值就是ps[t]的`[3][0]`元素。

3. 后向过程，从seq_lengh到0的展开。利用前向过程的结果向量ps, hs以及输入xs

也是先看完整代码:

```python
dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
```

**细节待增加，即计算这些梯度**

### Lua/Torch版更高效代码

训练了几个例子来证明多层LSTM的牛逼。可以学到很深的结构。

* Paul Ganham 生成器（画家与黑客作者）:2层，512神经元，350万参数，每层0.5的dropout，1M训练数据
* Shakespeare: 3层，512神经元，4.4M训练数据
* Wikipedia: 使用[文章](http://arxiv.org/abs/1308.0850)的网络，96M训练数据
* Latex结构: 16M训练数据
* Linux代码: 474M训练数据, 3层LSTM, 1千万参数
* 生成Baby名字: 8000训练数据输入，产生新名字。有强大的启发作用

## 进一步理解RNN的训练

### 训练中是如何进化的

通过每隔100个迭代打印生成的文本，我们发现模型先学会词和空格的结构，然后是单词，从短单词到长单词，最后是主题等跨越多个单词的存在。

### 可视化RNN

经过观察发现，不同的cell负责不同的任务，有些是负责识别URL的位置，有些是负责识别markdown符号\[\[ \]\]的位置。



## 参考
[1] http://karpathy.github.io/2015/05/21/rnn-effectiveness/  
[2] http://cs231n.github.io/neural-networks-case-study/#grad  
