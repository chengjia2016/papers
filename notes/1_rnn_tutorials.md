# RNN及LSTM零基础教程

## 一 教程分为如下部分

1. 介绍RNN
2. 用Theano实现RNN
3. 理解BPTT(BackPropagation Through Time)算法和梯度消失问题
4. 实现GRU/LSTM RNN

本教程会实现一个基于RNN的语言模型，这个语言模型的第一个用途可以用来计算任意一句话出现在真实世界中的概率，这种模型一般会用于机器翻译系统。  
第二个用途就是产生新的句子（这个更酷一些）。比如让RNN模型学习了莎士比亚的写作风格后，就能写出类似莎士比亚的句子。

就让我们一步一步开学学习吧。

## 二 什么是RNN呢?

RNN主要是利用序列信息。在传统神经网络中的输入是彼此独立的（输出也是）。但这种独立对于很多任务来说是不利的。如果你想预测一个句子中的下一个单词是什么，最好是知道这个单词之前有哪些单词。RNN中的R是recurrent, 是周期性, 反复的意思, 表示会对句子中的每一个单词用相同的方法处理，让输出基于之前的计算结果。理论上RNN可以利用任意长度的句子，但实际中我们只关注最近的几步。一个典型的RNN示意图如下:

![rnn](images/rnn.jpg)

上图是将左侧的RNN展开成完整的网络。展开的意思就是把需要关注的序列对应的网络写出来。比如对于一个句子，我们只关心5个单词，那么RNN网络就被展开成一个5层的神经网络，每一层对应序列中的一个单词。网络中的一些规则定义如下:

* $x_t$表示在t时刻(step)的输入。比如$x_1$是句子中第二个单词对应的one-hot向量(向量中只有1个1，其余都是0)。
* $s_t$是t时刻(step)的隐状态，是网络中作为"记忆"的存在。$s_t$是基于前一个隐状态和当前的输入计算出来的: $s_t=f(Ux_t+W{s_{t-1}})$ 。$f()$通常是非线性函数（如$tanh$, $ReLU$)。初始隐状态$s_{-1}$用于计算第一个隐状态，通常$s_{-1}$被初始化为全0。
* $o_t$是t时刻(step)的输出。比如要预测句子的下一个单词是什么，$o_t$就我们词典中每一个单词出现的概率，$o_t=softmax(Vs_t)$

有些地方需要注意下:

* $s_t$可以被理解为网络的记忆。$s_t$用于捕捉之前的所有时刻发生了啥。本时刻的输出$o_t$完全取决于t时刻的记忆。在实际中会很复杂，因为$s_t$不能捕捉太多之前的步骤。
* 不同于传统的深度神经网络（每层拥有不同的参数），RNN在所有层间（又叫step，跟上面的时刻对应)共享参数$(U, V, W)$ 。实际上我们是在各层执行相同的事情，只是输入不同罢了。这就极大的减少了我们需要学习的参数个数。
* 上面的图中每个step都有输出，但是在实际中不一定每个step都需要输出。比如我们想预测一个句子的感情色彩，我们只关系这个句子的最后输出。同样的，也不是每个step上都需要有输入。RNN的主要特点就在于其隐状态，是用于捕获一些具有顺序特点的信息而已。

## 三 RNN可以做什么呢？

很多NLP任务都可以利用RNN。RNN中最常用的是LSTMs，这个比原始RNNs在捕获有长依赖信息时更好用。LSTMs与RNN从本质上相同，只是在计算隐状态时不同而已。下面是一些RNN的应用例子。

### 1 语言模型以及生成文字

给出一系列的词语，我们希望预测在给定前一个词语时下一个词语的概率。语言模型给了我们可以评估一个句子确实是一个句子的尺子，这在机器翻译中很重要，因为高概率的句子通常是正确的结果。预测下一个词的副产品就是我们得到了一个生成模型，即允许我们生成新的文本。并且根据我们喂给模型的训练数据，几乎可以生成任何相应特点的新句子。在语言模型中个，输入一般是一系列的词（编码成one-hot形式的向量），输出就是预测的词。在训练时，我们设置$o_t=x_{t+1}$，因为在t step时的真正输出是$x_{t+1}$。

### 2 机器翻译

![机器翻译](images/language_translation.png)

机器翻译跟语言模型很相似，输入都是词语的序列，比如原文是德文，我们希望输出是英文。关键的不同是输出是在我们见过了所有的输入后才有的，因为翻译后的句子的第一个词是需要了解了整个句子原文的。

### 3 语音识别(Speech recognition)

输入声音信号，预测如何根据语音将他们转换成文字。

### 4 生成图片描述

与CNN一起使用, RNN模型作为整体模型的一部分生成未标注图片的描述。组合而成的整体模型可以将从图片里面找到的物品和文字一一对应起来。

![captioning](images/captioning.png)

图片来源是neuraltalk的大牛Karpathy那里借的:  
<http://cs.stanford.edu/people/karpathy/deepimagesent/>

## 四 训练RNN

训练RNN与训练传统NN类似。同样使用BP算法, 但有一些小变化。因为网络中各层（time steps）共用相同的参数，因此每层输出的梯度不仅与当前的层有关，还有之前的各层。比如要计算t＝4的梯度，我们需要前面3层的梯度，把他们加起来。这个叫做BPTT`Backpropagation Through Time`。但是原始RNN使用BPTT训练长依赖关系时时效果很不好，因为著名的`消失的梯度`问题。LSTM就是解决这个训练长依赖时的难题被设计出来的。

后面会有训练RNN的具体方法及代码。

## 五 RNN的扩展

经年累月，伟大的研究者开发了更加复杂的RNN去解决原生RNN的短处。简单介绍下先。

### 1 双向RNN

主体思想是t时刻输出不仅仅依赖于之前的步骤，还与未来有关。比如想要预测一句话中缺失的单词，考虑句子的上下文显然是更加有效的。双向RNN很简单，仅仅是两个RNN摞在一起，而输出就是基于两个RNN的隐状态。

![](images/bidirectional-rnn.png)

### 2 深度双向RNN

与双向RNN类似，区别仅仅是在每层中有多个子层而已。实践中这个模型学习能力更好，但也需要更多的训练数据。

![](images/deep_bi_rnn.png)

### 3 LSTM网络

LSTM最近大热。它与RNN并没有本质的区别，仅仅区别于计算隐状态的方法。LSTM中的记忆体被称作cells，可以被理解为一个黑盒，输入是前一个状态$h_{t-1}$和当前的输入$x_t$。cells内部决定记住哪些（以及删掉哪些）。因此它可以将之前的状态，当前的记忆，以及当前的输入合并在一起。在实践中，这样的cells可以很好的捕获长依赖关系。

## 六 语言模型

在本教程里会用RNN实现一个语言模型。加入我们现在有一个句子，里面有m个单词。语言模型就是预测这个句子出现的可能性，定义如下:

$
P(w_1,...,w_m)=\displaystyle\prod_{i=1}^{m} P(w_i|w_1,...w_{i-1})
$

理解为句子出现的概率是组成其的单词在此单词前面单词出现的情况下出现的概率的乘积（好绕）。比如"He want to buy some chocolate"出现的概率是在给定"He want to buy some"的前提下"chocolate"出现的概率，乘以给定"He want to buy"的前提下"some"出现的概率，以此类推直到第一个单词"He"出现的概率。

为啥这么定义？为啥给一个句子出现的概率？

首先这个语言模型可以被用作一个评分标准。比如一个机器翻译系统对于输入产生了多个候选翻译结果，此时就可以通过语言模型选出出现可能性最高的句子。同样的原理可以应用在语音识别中。

在得到语言模型的同时会得到一个副产品。因为模型可以根据已经前序的单词预测下一个单词出现的概率，我们可以利用此生成新文本，即得到一个生成模型。通过给定已经存在的前序单词序列，我们不断预测下一个出现的单词，直到最终得到一个完整的句子。大神Andrej Karparthy的[demo](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)展示了此能力。通过基于单个字符的训练，可以生成莎士比亚的句子，甚至Linux代码...

语言模型的数学定义是每一个单词的概率是基于所有前序单词。在实际中，鉴于计算和存储的限制很难表示此种长依赖模型，而是仅仅关心当前单词之前的有限个单词。RNNs理论上可以捕获长依赖模型，但会很复杂。

## 七 训练数据和预处理

我们需要文本数据去训练语言模型。幸运的是只需要原始的文本，而不是标注数据。下载15,000个reddit上的比较长的评论用于训练，然后我们的模型生产出的句子就会类似于reddit上的用户写出的评论（希望如此哈）。但再此之前，需要做一些前期工作把原始数据变成可用的格式。

### 1 分词

因为我们希望预测是基于单个单词的，所以我们需要把句子先分成单词。如果简单的根据空格切分句子会无法处理标点符号，比如"He left!"应该被分成"He","left","!"。这个工作交给[NLTK(Natural Language Toolkit)](http://www.nltk.org/)工具去做，利用其`word_tokenize`和`sent_tokenize`方法。

### 2 去掉低频词

训练数据中某些单词仅仅出现一两次，最好把此类词删掉。因为一个巨大的词库会让训练变慢(后面会说为啥变慢)，另外因为这些低频词没有足够多的训练样本，即便参与训练也不能学会正确使用其的方法，就跟人类真正学会使用一个单词时需要在多种上下文中见过他一样。

我们通过参数`vocabulary_size`来限制最多有多少单词参与训练(初始设置成8000,随便修改试试)。然后把所有不在此词库中的单词都用`UNKNOWN_TOKEN`替代。比如单词"nonlinearities"没有被包括在词典中，那么句子"nonlinearities are important in neural networks"会变成"UNKNOWN_TOKEN are important in neural networks"。UNKNOWN_TOKEN会被加入到词典中如其他单词一样，并参与训练和预测。当在生成句子时，我们同样替换UNKNOWN_TOKEN，比如随机选取一个不在词典中的单词，或者重新预测直到得到一个没有UNKNOWN_TOKEN的句子。

### 3 句子开头与结束标记

我们关心什么样的次会是句子的开头或者结尾。为了解决此问题，将SENTENCE_START放置在句子开头，SENTENCE_END放置在句子结尾。这样就可以学到哪些单词最可能是句子的开头。

### 4 构建训练数据矩阵

RNN的输入是向量，而不是字符串。因此要做一个单词到数字的映射, `index_to_word`和`word_to_index`。比如单词"friendly"映射成2001。训练样本`x`是这样的`[0, 179, 341, 416]`, `0`表示SENTENCE_START。对应的结果`y`是`[179,341,416,1]`。记住我们的目标是预测下一个单词，所以`y`向量就是`x`向量左移一位并在最后加上SENTENCE_END标记。比如179的预测就是341。

```python
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
import csv
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
  reader = csv.reader(f, skipinitialspace=True)
  reader.next()
  # Split full comments into sentences
  sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
  # Append SENTENCE_START and SENTENCE_END
  sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parse %d sentences." % (len(sentences))

#Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

#Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i, w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown tokens
for i, sent in enumerate(tokenized_sentences):
  tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" %s tokenized_sentences[0]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

```

训练数据这个样子：

```
x:
SENTENCE_START what are n't you understanding about this ? !
[0, 51, 27, 16, 10, 856, 53, 25, 34, 69]

y:
what are n't you understanding about this ? ! SENTENCE_END
[51, 27, 16, 10, 856, 53, 25, 34, 69, 1]
```

## 八 构建RNN

再看一下这个图，RNN网络, 以及根据计算步骤在前向传播时展开的样子。

![rnn](images/rnn.jpg)

让我们看看RNN语言模型实际长什么样。输入$x$是单词序列，每个$x_t$是一个单词。因为我们矩阵乘法不能直接使用单词对应的序列号(比如某单词对应序号36)。因此我们用维度为vocabulary_size的向量以one-hot的形式表示一个单词。比如36序号的单词的one-hot向量就是在第36的位置是1，其余位置都是0。$x_t$对应一个向量，则$x$就是一个数组了，每一行代表一个单词。这个转换会在神经网络的代码中实现，而不在预处理阶段。网络的输出$o$有相同的格式。$o_t$是vocabulary_size维度的向量，每一维上的元素表示那个位置表示的单词作为句子预测的下一个单词的概率。

回忆RNN的公式:

$s_t = tanh(Ux_t+Ws_{t-1})$
$o_t = softmax(Vs_t)$

写出矩阵和向量的维度总是有助于理解模型。假设字典维度是8000, C=8000, 隐层尺寸是100, H=100。隐层尺寸理解为网络的记忆体。尺寸越大可以学到更复杂的模式，但需要更多的计算:

$x_t \in \mathbb{R}^{8000},o_t \in \mathbb{R}^{8000}$
$s_t \in \mathbb{R}^{100}$
$U \in \mathbb{R}^{100*8000}$
$V \in \mathbb{R}^{8000*100}$
$W \in \mathbb{R}^{100*100}$

$U,V,W$是我们模型的参数，是要从数据中学出来的。因此我们总共需要学出$2HC+H^2$个参数($2HC$是$U,V$的个数, $H^2$是$w$的个数)。在C=8000, H=100下，总共参数是1,610,000, 百万级的参数。维度展示可以告诉我们模型的瓶颈在哪里。观察$Ux_t$，$x_t$是one-hot向量, 与$U$相乘时实际只是选择$U$中的一列，不需要实际执行矩阵乘法。因此，整个模型中计算量最大的地方就是计算$Vs_t$了，这就是我们需要限制字典大小的实际原因。

有了上面的铺垫，现在开始实现。

### 1 初始化

因为后面会有用Theano实现的版本, 我们称当前版本为**RNNNumpy**。初始化$U,V,W$有一些技巧的，不能简单的都初始化成0，这会导致网络很难收敛。我们需要随机初始化参数。因为初始化的方法会影响训练结果，这个领域也有很多研究。一般来说最佳的初始化方法依赖激活函数的选择(我们这里是$tanh$), 建议是从范围$[-\frac{1}{\sqrt{n}},\frac{1}{\sqrt{n}}]$内取随机值初始化，$n$是上一层到本层的连接个数。其实只要用很小的数初始化，一般来说都会工作良好。

```python
class RNNNumpy:
  def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
    # Assign instance variables
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    # Randomly initialize the network parameters
    self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
    self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
    self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden), (hidden_dim, hidden_dim))
```

`word_dim`是字典大小, `hidden_dim`是隐层宽度。`bptt_truncate`后面会说。

### 2 前向传播

现在实现前向传播，用于预测下一个次的出现概率。根据这个上面出现过的公式

$s_t = tanh(Ux_t+Ws_{t-1})$
$o_t = softmax(Vs_t)$

```python
def forward_propagation(self, x):
  # The total number of time steps
  T = len(x)
  # During forward propagation we save all hidden states in s because need them later.
  # We add one additional element for the initial hidden, which we set to 0
  s = np.zeros((T + 1, self.hidden_dim))
  s[-1] = np.zeros(self.hidden_dim) # I think no use this line
  # The outputs at each time step. Again, we save them for laster.
  o = np..zeros((T, self.word_dim))
  # For each time step...
  for t in np.arange(T):
    # Note that we are indexing U by x[t]. This is the same as multiplying U with a one-hot vector.
    s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
    o[t] = softmax(self.V.dot(s[t]))
  return [o, s]

RNNNumpy.forward_propagation = forward_propagation
```

我们不仅返回了计算结果，还有隐状态。后面计算梯度时会用到它们，这里返回可以避免后续重复计算。$o$中每一个元素$o_t$表示一个单词的概率，有时我们只关心概率最高的那个词，比如在评估模型时。**predict** 函数用来做这个:

```python
def predict(self, x):
  # Perform forward propagation and return index of the highest score
  o, s = self.forward_propagation(x)
  return np.argmax(o, axis = 1)

RNNNumpy.predict = predict
```

尝试下新实现的函数:

```python
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10])
print o.shape
print o
```

输出:

```
(45, 8000)
[[ 0.00012408  0.0001244   0.00012603 ...,  0.00012515  0.00012488
   0.00012508]
 [ 0.00012536  0.00012582  0.00012436 ...,  0.00012482  0.00012456
   0.00012451]
 [ 0.00012387  0.0001252   0.00012474 ...,  0.00012559  0.00012588
   0.00012551]
 ...,
 [ 0.00012414  0.00012455  0.0001252  ...,  0.00012487  0.00012494
   0.0001263 ]
 [ 0.0001252   0.00012393  0.00012509 ...,  0.00012407  0.00012578
   0.00012502]
 [ 0.00012472  0.0001253   0.00012487 ...,  0.00012463  0.00012536
   0.00012665]]
```

对于句子中的每个单词（上面是45个单词），每个单词都有计算词库中每个词作为其下一个的概率。下面几行代码是取的句子每个单词下一个是词库里哪个单词（概率最高那个）:

```python
predictions = model.predict(X_train[10])
print predictions.shape
print predictions
```

输出:

```
(45,)
[1284 5221 7653 7430 1013 3562 7366 4860 2212 6601 7299 4556 2481 238 2539
 21 6548 261 1780 2005 1810 5376 4146 477 7051 4832 4991 897 3485 21
 7291 2007 6006 760 4864 2182 6569 2800 2752 6821 4437 7021 7875 6912 3575]
```

### 3 计算损失函数

模型训练需要明确如何度量error，即损失函数$L$。我们的训练目标就是寻找在给定训练数据集合上，最小化$U,V,W$的损失函数$L$。通常损失函数用`cross-entropy loss`。如果有N个训练数据样本, C个分类(字典大小), 预测值$o$与真实值$y$的**loss**定义为:

$L(y,o) = -\frac{1}{N}\displaystyle\sum_{n \in N}y_nlogo_n$

cross-entropy loss实际上就是把所有训练数据上预测输出和真实值的差累加在一起。loss越大则表示$y$和$o$的差距越大。**calculate_loss** 实现如下:

```python
def calculate_total_loss(self, x, y):
  L = 0
  # For each sentence...
  for i in op.arange(len(y)):
    o, s = self.forward_propagation(x[i])
    # We only care about our prediction of the "correct" words
    correct_word_predictions = o[np.arange(len(y[i])), y[i]]
    # Add to the loss based on how off we are
    L += -1 * np.sum(np.log(correct_word_predictions))
  return L

def calculate_loss(self, x, y):
  # Divide the total loss by the number of training examples
  N = np.sum((len(y_i) for y_i in y))
  return self.calculate_total_loss(x,y) / N

RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss
```

思考下对于随机的预测，loss应该是什么，这会给我们一个基准用于评判我们实现的代码是不是正确的。因为我们有C个分类（字典大小），每个词被预测准确的概率就是$1/C$，loss应该是:$L=-\frac{1}{N}Nlog{\frac{1}{C}}=logC$。

```python
# Limit to 1000 examples to save time
print "Expected loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])
```

输出:
```
Expected Loss for random predictions: 8.987197
Actual loss: 8.987440
```

计算整个数据集上的loss是个很费计算的操作，如果数据集大可能会消耗数个小时。

### 4 利用SGD和BPTT训练RNN

我们目标是寻找到可以最小化loss的一组参数$U, V, W$。通常的方法是使用SGD。SGD的思想很简单，在观察每一条训练数据后到让参数朝着可以减少error的方向变化。这个方向就是loss对几个参数的梯度: $\frac{\partial{L}}{\partial{U}},\frac{\partial{L}}{\partial{V}},\frac{\partial{L}}{\partial{W}}$。SGD还需要**学习率**，即希望每次朝着梯度的方向变化多少。SGD虽然简单，但是想要优化好SGD却不是容易的事，可以参考[cs231的这篇文章](http://cs231n.github.io/optimization-1/)。我们这里只会实现一个简单的SGD用于理解。

那么到底怎么实现计算梯度呢？在传统神经网络中我们使用Backpropagation算法, 在RNN我们使用其简单变种BPTT。网络中的参数在各个步骤(time steps)共享，每个步骤输出的梯度不仅依赖于当前时间点，还依赖于之前的时间点。更加一般的backpropagation可以参考[Olah的文章](http://colah.github.io/posts/2015-08-Backprop/)和[cs231的文章](http://cs231n.github.io/optimization-2/)。简单的可以吧BPTT理解为黑盒，输入是训练数据(x,y),输出是各个参数的梯度$\frac{\partial{L}}{\partial{U}},\frac{\partial{L}}{\partial{V}},\frac{\partial{L}}{\partial{W}}$

```python
def bptt(self, x, y):
  T = len(y)
  # Perform forward propagation
  o, s = self.forward_propagation(x)
  # We accumulate the gradients in these variables
  dLdU = np.zeros(self.U.shape)
  dldV = np.zeros(self.V.shape)
  dldW = np.zeros(self.W.shape)
  delta_o = o
  delta_o[np.arange(len(y)),y] -= 1.
  # For each output backwards...
  for t in np.arange(T)[::-1]:
    dldV += np.outer(delta_o[t], s[t].T)
    # Initial delta calculation
    delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
    # Backpropagation through time (for at most self.bptt_truncate steps)
    for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
      # print "Backpropagation step t=%d bptt setp=%d " % (t, bptt_step)
      dldW += np.outer(delta_t, s[bptt_step-1])
      dldU[:, x[bptt_step]] += delta_t
      # Update delta for next step
      delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
  return [dLdU, dLdV, dLdW]

RNNNumpy.bptt = bptt
```

### 5 检查梯度

实现完BPTT后最好再实现下梯度检查，用于检查我们的实现是否正确。梯度检查就是检查下参数的导数是不是等于当前点的斜率，当前点的斜率可以通过微小的改变参数计算出来:

${\displaystyle} \frac{\partial{L}}{\partial{\theta}} \approx {\lim_{h\to0}\frac{J(\theta + h)-J(\theta-h)}{2h}}$

我们比较通过backpropagation计算出来值和上面公式的近似值，如果没有很大差异就对了。近似值需要计算所有参数的total loss, 因此梯度检查非常消耗计算(这里我有有百万级别的参数)。因此最好是现在小数据上测试下。

```python
def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    # Calculate the gradients using backpropagation. We want to checker if these are correct.
    bptt_gradients = self.bptt(x, y)
    # List of all parameters we want to check.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            gradplus = self.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            gradminus = self.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            # Reset parameter to original value
            parameter[ix] = original_value
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error &gt; error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)

RNNNumpy.gradient_check = gradient_check

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])
```

### 6 实现SGD

现在可以实现SGD。分成2步，1 **sgd_step** 计算梯度，然后更新参数。2 外循环遍历所有训练数据，并调整学习率

```python
# Performs one step of SGD.
def sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW

RNNNumpy.sgd_step = sdg_step
```

```python
# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
```

完成！让我们亲身体验下训练

```python
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
%timeit model.sgd_step(X_train[10], y_train[10], 0.005)
```

在我的macbook pro retina 15上，运行一次sgd_step需要76ms。我们训练数据中有80,000条数据，所以一轮迭代需要80000*0.076＝1.68个小时，进行多轮训练需要很多小时。我们还是在很小的数据集上做的实验，那么如果是大公司或者研究室内使用的大数据集咋办？

幸运的是有很多办法可以加速训练，比如优化代码，修改模型，使用hierachical softmax, 增加projection层避免大的矩阵乘法。我们这里尝试使用GPU来加速下。在此之前，先在超小数据集上看看实现是否正确。

```python
np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy(vocabulary_size)
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
```

输出:
```
2016-08-22 14:46:26: Loss after num_examples_seen=0 epoch=0: 8.987425
2016-08-22 14:46:35: Loss after num_examples_seen=100 epoch=1: 8.976270
2016-08-22 14:46:44: Loss after num_examples_seen=200 epoch=2: 8.960212
2016-08-22 14:46:53: Loss after num_examples_seen=300 epoch=3: 8.930430
2016-08-22 14:47:02: Loss after num_examples_seen=400 epoch=4: 8.862264
2016-08-22 14:47:12: Loss after num_examples_seen=500 epoch=5: 6.913570
2016-08-22 14:47:21: Loss after num_examples_seen=600 epoch=6: 6.302493
2016-08-22 14:47:30: Loss after num_examples_seen=700 epoch=7: 6.014995
2016-08-22 14:47:39: Loss after num_examples_seen=800 epoch=8: 5.833877
2016-08-22 14:47:48: Loss after num_examples_seen=900 epoch=9: 5.710718
```

看着loss在下降，说明代码实现是对的。

## 九 利用Theano和GPU训练

将所有numpy计算的地方改成Theano的实现, 定义了**RNNTheano**。瞬间得到加速。用Theano的cpu模式会比numpy的版本块，用Theano的gpu模式会更快。这样的加速很有用，可以让训练时间从1个星期缩减到1天左右。

为了避免长时间的训练，我提供了一个训练好的模型用于体验，隐层使用50个维度，词典是8000。训练50轮，用了约20小时。loss还可以再降，训练更久可以得到更好的结果。用下面的函数加载模型:

```python
from utils import load_model_parameters_theano, save_model_parameters_theano

model = RNNTheano(vocabulary_size, hidden_dim=50)
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
# save_model_parameters_theano('./data/trained-model-theano.npz', model)
load_model_parameters_theano('./data/trained-model-theano.npz', model)
```

## 十 生成文本

现在已经有了模型，写个helper函数通过模型生成句子吧:

```python
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) &lt; senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)
```

这些是生成的句子（挑选了一些比较好的）:
* Anyway, to the city scene you’re an idiot teenager.
* What ? ! ! ! ! ignore!
* Screw fitness, you’re saying: https
* Thanks for the advice to keep my thoughts around girls.
* Yep, please disappear with the terrible generation.

看这些生成的句子，模型可以学到使用语法，可以使用标点符号。  
但是更多的句子是没有意义的，而且有语法错误。原因之一是我们训练的时间不够长，或者用的训练数据不够多。但是更主要的原因是，原生的RNN不能产生有意义的文字是因为它不能够学到单词间较长距离的依赖关系。这就是RNN再被发明出来后没有被普遍使用的原因，理论上很好，但是实践不佳。LSTM解决了这个问题，让我们继续往下看。








## 参考资料

[1] <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>  
[2] <http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/>
