# 无网编译tf注意事项

本地编译主要修改两个文件:  
1. WORKSPACE  
2. tensorflow/workspace.bzl  
以及编译crosstool工具配置修改  

更新到最新版本的tensorflow

workspace.bzl有变动

依赖更新

去掉path_prefix

注意strip_prefix, 有些依赖对应的external中的目录path需要调整

先通过本地CPU编译

IDC上GPU编译

swig依赖, 否则configure无法看到cuda

config编译配置:
1. cuda sdk: 7.5
2. cuda 7.5 toolkit: default, /usr/local/cuda
3. cudnn: 5
4. cudnn lib: default, /usr/local/cuda
5. compute capability: 3.7

在IDC上无法找到gcc include相关依赖问题, 新版本的修改与以前有所不同
修改:/tensorflow/third_party/gpus/cuda_configure.bzl
通过hack的方法，加入依赖

```python
def get_cxx_inc_directories(repository_ctx, cc):
  """Compute the list of default C++ include directories."""
  result = repository_ctx.execute([cc, "-E", "-xc++", "-", "-v"])
  index1 = result.stderr.find(_INC_DIR_MARKER_BEGIN)
  if index1 == -1:
    return []
  index1 = result.stderr.find("\n", index1)
  if index1 == -1:
    return []
  index2 = result.stderr.rfind("\n ")
  if index2 == -1 or index2 < index1:
    return []
  index2 = result.stderr.find("\n", index2 + 1)
  if index2 == -1:
    inc_dirs = result.stderr[index1 + 1:]
  else:
    inc_dirs = result.stderr[index1 + 1:index2].strip()

  return [repository_ctx.path(_cxx_inc_convert(p))
          for p in inc_dirs.split("\n")] + ["/usr/include/", "/include/",
          "/lib/gcc/x86_64-redhat-linux/4.8.2/include/",
          "/usr/lib/gcc/x86_64-redhat-linux/4.8.2/include/"] ## hack here
```
