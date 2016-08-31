# 制作neuraltalks2的docker image

## docker基本命令

### 创建容器:

```
docker run -it -d --net=host --name='willwywang_dev' --privileged=true -v /data1:/data1 -v /dev:/dev  -v /cephfs:/cephfs docker.oa.com:8080/datamining/mariana2.0_cuda7.5:devenv /bin/bash
```

### 进入某个正在运行的容器:

`docker exec -it willwywang_dev /bin/bash`

### 查看本地有哪些镜像

`docker images`

### 查看当前有哪些容器在运行中

`docker ps [-a]`

### 删除某个容器

先停止, 再删除  
`docker stop container_id`  
`docker rm container_id`  

### 从仓库拉取某个镜像

`docker pull datamining/mariana2.0_cuda7.5:devenv`

### 搜寻某个镜像

`docker search xxx`

### 固化(提交)容器成为镜像

`docker commit -m "comment" container_id [repo:tag]`

### 把某个镜像提交到仓库

`docker push repo:tag`

## 创建neuraltalks2的images

1. 创建container
2. [按照内网环境搭建neuraltalk](http://wangyang.online/6_neuraltalk_compile_without_network.html)
3. 固化container成为image, 并提交

```
> docker commit -m "for image captioning dev" e73d6f446fd1  datamining/mariana2.0_cuda7.5:image_caption_dev

> docker images #检查是否有这个image

> docker push docker.oa.com:8080/datamining/mariana2.0_cuda7.5:image_caption_dev # oa-auth with pin code

# image 位置: datamining/mariana2.0_cuda7.5/tags/image_caption_dev
```


4. 使用此image进行测试 (TODO)
