# 无联网环境下搭建neuraltalk2

公司的机房里面是不能联网的, 这就需要把所有依赖都下载下来手动安装所有依赖。  
这里记录下在其搭建过程，以备查阅。

**注: 内网环境具备了yum和pip, 主要是luarocks的安装需要手动**

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
yum install protobuf-compiler protobuf
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
d cudnn-7.5-linux-x64-v5.0-ga/cuda/lib64
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

## 训练
