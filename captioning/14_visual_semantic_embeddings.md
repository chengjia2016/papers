# Visual Semantic Embeddings

论文: Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models

对于图文对, 把文字用LSTM进行embedding, 从CNN中得到的图片特征映射到LSTM的隐状态embedding空间。

优化的目标:  
A pairwise ranking loss is minimized in order to learn to rank images and their descriptions.

encoder用于rank 图和文字
decoder用于产生新的caption文字
