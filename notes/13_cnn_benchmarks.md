# CNN benchmarks
Ref: https://github.com/jcjohnson/cnn-benchmarks

# GPU卡的性能
Pascal Titan X > GTX 1080 > Maxwell Titan X  
TeslaM40还是Maxwell平台
Pascal Titan X Tflops有10, GTX1080 8.8, M40 7

# CNN模型能力
ResNet-50 好于 VGG-16  
ResNet-101 跟 VGG-19速度差不多, 但比VGG-16准确度好很多  
ResNet-200 最好,但最慢  
VGG-16跟VGG-19差不多  
**ResNet-101性价比最好**  

# torch下训练要用cuDNN, 而不是nn, 快很多
2x-3x

# GPU配合上cuDNN要比CPU快非常多
49x-74x
