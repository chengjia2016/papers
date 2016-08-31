# 无联网环境下搭建neuraltalk2

公司的机房里面是不能联网的, 这就需要把所有依赖都下载下来手动安装所有依赖。  
这里记录下在其搭建过程，以备查阅。

**注: 内网环境具备了yum和pip(需要配置), 主要是luarocks的安装需要手动**

<!-- more -->

## 从本地拷贝torch和neuraltalk2到IDC机器

torch.tgz  
neuraltalk2.tgz

## torch安装

### bash install-deps

因操作系统非外部标准版本, 直接运行`bash install-deps`会提示操作系统版本不支持。  
因此需要手动安装依赖。IDC系统类似CentOS, 可以参考centos的依赖进行安装:


```bash
sudo yum install -y make cmake curl readline-devel \
ncurses-devel \
gcc-c++ gcc-gfortran git gnuplot unzip \
libjpeg-turbo-devel libpng-devel \
ImageMagick GraphicsMagick-devel fftw-devel \
sox-devel sox zeromq3-devel \
qt-devel qtwebkit-devel sox-plugins-freeworld
```

```bash
pip install ipython
```

### install.sh

```
clean.sh # 别处copy过来的torch目录要先clean
install.sh
```

进行编译  

编译完成后输入`yes`允许更新`/root/.bashrc`  

激活环境变量  

```
source ~/.bashrc
```

运行`th`检查安装是否成功, 成功如下

```
[root ~/torch]# th

  ______             __   |  Torch7
 /_  __/__  ________/ /   |  Scientific computing for Lua.
  / / / _ \/ __/ __/ _ \  |  Type ? for help
 /_/  \___/_/  \__/_//_/  |  https://github.com/torch
                          |  http://torch.ch

th>
```

## 安装neuraltalk2依赖

安装顺序:

### cjson

```
luarocks make lua-cjson-2.1.0-1.rockspec

```

### loadcaffe

依赖protobuf
```
yum install protobuf-compiler protobuf-devel protobuf
```

```
rm -rf build # 清理上次cmake结果
luarocks make loadcaffe-1.0-0.rockspec
```

### hdf5

```
yum install hdf5 hdf5-devel
```

依赖totem
```
luarocks make rocks/totem-0-0.rockspec
```

torch-hdf5
```
luarocks make hdf5-0-0.rockspec
```

h5py
```
pip install h5py
```

## 安装gpu相关依赖

### cutorch

```
luarocks make rocks/cutorch-scm-1.rockspec
```

### cunn

```
luarocks make rocks/cunn-scm-1.rockspec
```


### cudnn

先下载`cudnn-7.5-linux-x64-v5.0-ga.tgz`, 确保cudnn的lib64和include都安装到/usr/local/中, [参考安装tensorflow时cudnn相关的操作](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#prepare-environment-for-mac-os-x)

```
luarocks make cudnn-scm-1.rockspec
```

## 运行evaluation

GPU模式
```
th eval.lua -model model/model_id1-501-1448236541.t7 -image_folder images/ -num_images 10
```

### 如果运行时出错, 提示cudnn的版本有问题  
是因为原来tensorflow用的是v4, 我们要替换成v5

```
/usr/local/cuda/lib64
-rwxr-xr-x  1 59M Aug 17 18:09 libcudnn.so
-rwxr-xr-x  1 59M Aug 17 18:09 libcudnn.so.4
-rwxr-xr-x  1 59M Aug 17 18:09 libcudnn.so.4.0.7
-rw-r--r--  1 60M Aug 17 18:09 libcudnn_static.a
```

现在需要替换成
```
cd cudnn-7.5-linux-x64-v5.0-ga/cuda/lib64
lrwxrwxrwx 1 13 Apr 23 10:52 libcudnn.so -> libcudnn.so.5
lrwxrwxrwx 1 17 Apr 23 10:52 libcudnn.so.5 -> libcudnn.so.5.0.5
-rwxrwxr-x 1 59909104 Apr 23 08:15 libcudnn.so.5.0.5
-rw-rw-r-- 1 58775484 Apr 23 08:15 libcudnn_static.a
```

另外原来tensorflow的cudnn.h也需要替换, 要配套使用
```
cd /usr/local/cuda/include
-r--r--r--  1 75330 Aug 17 18:09 cudnn.h
```

替换成
```
cd cudnn-7.5-linux-x64-v5.0-ga/cuda/include
-r--r--r-- 1 99370 Apr 23 02:24 cudnn.h
```

把原来的备份好
```
/usr/local/cuda/include/cudnn.h
/usr/local/cuda/lib64/libcudnn.so
/usr/local/cuda/lib64/libcudnn.so.4
/usr/local/cuda/lib64/libcudnn.so.4.0.7
/usr/local/cuda/lib64/libcudnn_static.a
```
备份到/usr/local/cuda/lib64/cudnn_v4  和/usr/local/cuda/include/cudnn_v4  
使用新的cudnn-7.5-linux-x64-v5.0-ga中的.h和.so

**替换后重新安装cudnn-scm-1.rockspec**

### 如果提示解析JPEG错误

```
DataLoaderRaw loading images from folder:       images/
listing all images in directory images/
DataLoaderRaw found 30 images
constructing clones inside the LanguageModel
images/._004545.jpg
/root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/image/init.lua:221: Not a JPEG file: starts with 0x00 0x05
stack traceback:
[C]: in function 'load'
/root/torch/install/share/lua/5.1/image/init.lua:221: in function 'loader'
/root/torch/install/share/lua/5.1/image/init.lua:369: in function 'load'
./misc/DataLoaderRaw.lua:82: in function 'getBatch'
eval.lua:116: in function 'eval_split'
eval.lua:173: in main chunk
[C]: in function 'dofile'
/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
[C]: at 0x004064f0
```

确认images目录里面有没有`._004545.jpg`类似的临时文件, 有则删掉

### 使用CPU模式正常, GPU模式错误

CPU模式

```
th eval.lua -gpuid -1 -model model/model_id1-501-1448236541.t7_cpu.t7 -image_folder images/ -num_images 10
```

GPU模式
```
/root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/nn/LookupTable.lua:73: bad argument #3 to 'index' (Tensor | LongTensor expected, got torch.CudaLongTensor)
stack traceback:
[C]: in function 'index'
/root/torch/install/share/lua/5.1/nn/LookupTable.lua:73: in function 'forward'
./misc/LanguageModel.lua:199: in function 'sample'
eval.lua:135: in function 'eval_split'
eval.lua:173: in main chunk
[C]: in function 'dofile'
/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
[C]: at 0x004064f0
```

出现这个错误是因为nn, nngraph没有安装好, 重新安装后解决:  

```
luarocks make rocks/nn-scm-1.rockspec
luarocks make nngraph-scm-1.rockspec
```

安装玩torch后, nn, nngraph, image是自动安装了的, 但是版本不符合要求, 需要再重新手动安装下这3个依赖。  

## 准备训练数据

参考[Macbook上OS/X搭建neuraltalks2(有网络)](http://wangyang.online/5_neuraltalk_compile.html)

```
python prepro.py --input_json coco/coco_raw.json --num_val 5000 --num_test 5000 --images_root coco/images --word_count_threshold 5 --output_json coco/cocotalk.json --output_h5 coco/cocotalk.h5
```

生成: `cocotalk.h5`, 23G。

## 训练

为了让训练中可以打印BLEU/METEOR/CIDEr分数, 需要把  
`https://github.com/tylin/coco-caption` 放到coco-caption目录

并加上参数 `-language_eval 1`

运行:  
`th train.lua -input_h5 coco/cocotalk.h5 -input_json coco/cocotalk.json -language_eval 1`

在默认训练参数下, 在COCO数据集上的1个epoch是7500次迭代，在非精调(fine tuning)下1轮迭代是1小时左右。loss在2.7附近, CIDEr在0.4附近。在70000次迭代时CIDEr达到0.6, loss在2.5左右。最终CIDEr会在0.7左右。
然后停止训练, 打开精调开关, `-finetune_cnn_after 0` ，使用`-start_from`指定上一阶段的模型继续训练, 在2天后, CIDEr会达到0.9附近。

默认使用的是gpu 0号, 其忽略了bashrc中的`export CUDA_VISIBLE_DEVICE=2,3`

使用nvidia_smi查看时, 确实占用了0号gpu

```
Mon Aug 29 17:48:50 2016
+------------------------------------------------------+
| NVIDIA-SMI 352.79     Driver Version: 352.79         |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 0000:05:00.0     Off |                    0 |
| N/A   72C    P0   145W / 149W |   3007MiB / 11519MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla K80           On   | 0000:06:00.0     Off |                    0 |
| N/A   38C    P0    74W / 149W |    470MiB / 11519MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla K80           On   | 0000:85:00.0     Off |                    0 |
| N/A   50C    P0    60W / 149W |    163MiB / 11519MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla K80           On   | 0000:86:00.0     Off |                    0 |
| N/A   38C    P0    73W / 149W |    163MiB / 11519MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

### 若训练时出错:

原因:`RuntimeError: could not open display`

具体

```
image 99416: steps earphones flows emirates kitchenware separate pork gripping diaper concentrates stethoscope brown lifts grouped steve allowed
image 132773: seen efficient refrigerator fan daughter less progress unused foilage boarders feast orchids curtain clothe wrapped reclining
evaluating validation performance... 3200/3200 (9.179654)
Traceback (most recent call last):
  File "myeval.py", line 6, in <module>
    from pycocotools.coco import COCO
  File "/root/neuraltalk2/coco-caption/pycocotools/coco.py", line 48, in <module>
    import matplotlib.pyplot as plt
  File "/usr/lib64/python2.7/site-packages/matplotlib/pyplot.py", line 97, in <module>
    _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
  File "/usr/lib64/python2.7/site-packages/matplotlib/backends/__init__.py", line 25, in pylab_setup
    globals(),locals(),[backend_name])
  File "/usr/lib64/python2.7/site-packages/matplotlib/backends/backend_gtkagg.py", line 10, in <module>
    from matplotlib.backends.backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
  File "/usr/lib64/python2.7/site-packages/matplotlib/backends/backend_gtk.py", line 13, in <module>
    import gtk; gdk = gtk.gdk
  File "/usr/lib64/python2.7/site-packages/gtk-2.0/gtk/__init__.py", line 64, in <module>
    _init()
  File "/usr/lib64/python2.7/site-packages/gtk-2.0/gtk/__init__.py", line 52, in _init
    _gtk.init_check()
RuntimeError: could not open display
/root/torch/install/bin/luajit: ./misc/utils.lua:17: attempt to index local 'file' (a nil value)
stack traceback:
        ./misc/utils.lua:17: in function 'read_json'
        ./misc/net_utils.lua:202: in function 'language_eval'
        train.lua:218: in function 'eval_split'
        train.lua:305: in main chunk
        [C]: in function 'dofile'
        /root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
        [C]: at 0x004064f0
```

**解决办法:**

修改`coco-caption/pycocotools/coco.py`, 把`#import matplotlib.pyplot as plt`注释掉, 把`plt`相关的也注释掉。因为IDC服务器是没有display的。

### 精调模型

```
th train.lua -input_h5 coco/cocotalk.h5 -input_json coco/cocotalk.json -language_eval 1 -gpuid 2 -finetune_cnn_after 0 -start_from checkpoint/model_id.t7 -checkpoint_path checkpoint > log 2>&1 &
```

### train.lua的更多参数

#### 输入设置
- -input_h5: 默认 coco/data.h5, 训练数据
- -input_json: coco/data.json, 训练数据元数据
- --cnn_proto: model/VGG_ILSVRC_16_layers_deploy.prototxt, Caffe格式的CNN模型prototxt, 现在只能是VGGNet-16
- --cnn_model: model/VGG_ILSVRC_16_layers.caffemodel, Caffe格式的CNN模型
- -start_from: "", 用于指定checkpoint模型, 用于继续训练, eg. checkpoint/model_id.t7

#### 模型设置
- -rnn_size: 默认512, 用于指定RNN神经元个数
- -input_encoding_size: 512, 词语和图像编码后的大小

#### 优化: 通用
- -max_iters: 默认-1, 最大迭代轮数, -1表示无穷
- -batch_size: 16, 每个batch有多少张图片, batch对应句子个数就是batch_size*此图对应的句子个数
- -grad_clip: 0.1, 梯度截断阈值(一般小于5, 因梯度已经归一化过)
- -drop_prob_lm: 0.5, RNN语言模型dropout强度
- finetune_cnn_after: -1, -1表示不启动精调, 0表示0轮迭代立即启动精调
- seq_per_img: 5, 限制训练中每个图片生成句子的个数, 因为CNN前向计算很费时

#### 优化: RNN语言模型
- -optim: 默认adam, update方法, 可选rmsprop|sgd|sgdmon|adagrad|adam
- -learning_rate: 4e-4, 学习率
- -learning_rate_decay_start: -1， -1不衰减, 从哪次迭代开始衰减学习率
- -learning_rate_decay_every: 50000, 间隔多少迭代折半衰减学习率
- -optim_alpha: 0.8, alpha参数, 针对adagrad/rmsprop/momentum/adam
- -optim_beta: 0.999, beta参数, 针对adam
- -optim_epsilon: 1e-8, epsilon用于平滑时的分母

#### 优化: CNN模型
- -cnn_optim: 默认adam, CNN的优化方法
- -cnn_optim_alpha: 0.8, alpha for mementum
- -cnn_optim_beta: 0.999, beta for momentum
- -cnn_learning_rate: 1e-5
- -cnn_weight_decay: 0, L2衰减参数

#### 评估/保存
- -val_images_use: 默认3200, 周期的进行评估计算loss时用多少图片, -1是用所有
- -save_checkpoint_every: 2500, 多少迭代去存一个checkpoint
- -checkpoint_path: "", 保存checkpoint的路径, 空是当前目录
- -language_eval: 0, 是否进行language evaluation, 1=yes, 0=no, BLEU/CIDEr/METEOR/ROUGE_L, 需要coco-caption的代码
- -losses_log_every: 25, 多少轮去算一下losses, 用于决定是否dump

#### 其他
- -backend: cudnn, nn|cudnn
- -id: "", 用于标记本轮训练的id, 用于cross-val, 和用于dump文件
- -seed: 123, 随机数种子
- -gpuid: 0, 用哪个gpu, -1是用CPU

## 训练结果

目前训练还在进行中, 最新的Evaluation分数是:

```
validation loss:        2.4419743700307
{
  Bleu_1 : 0.642
  ROUGE_L : 0.473
  METEOR : 0.208
  Bleu_4 : 0.22
  Bleu_3 : 0.317
  Bleu_2 : 0.46
  CIDEr : 0.695
}
```

## 后续

* 了解Image Caption的几个评估指标意义
* 详解代码
* 构建训练数据, 用自有训练数据进行训练
* 把neuraltalks2制作成docker image, 便于后续使用
