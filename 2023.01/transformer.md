未加pos embedding

![Figure_1](C:\Users\Xsu1023\Desktop\Figure_1.png)



## ViT (ICLR'21)

![image-20230114013132734](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230114013132734.png)

将图片裁成若干16*16块后做cnn作为feature，喂给encoder后，output经过一个分类头



## ViLT (ICML'21)

![image-20230114012726466](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230114012726466.png)

多模态，只用transformer encoder，将word和cnn过的pic设为一个seq喂给encoder



## DETR (ECCV'20)

![image-20230114012622876](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230114012622876.png)

Image features: CNN处理原始图片后，将H*W拍平作为seq input，多通道视为特征维度

Object Queries: 100个锚框集合，记为object queries

最后output是关于这些query对应的锚框，两个分类头：class和bounding box分别计算锚框的两个loss



## MAE (CVPR'22)

![image-20230114022134546](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230114022134546.png)

![image-20230114022201873](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230114022201873.png)

decoder输入中被mask的块统一用一个共享可学习的向量来表示，加上位置信息

最后做任务时只用encoder，encoder类似于一个ViT



## GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph

![image-20230115003941733](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230115003941733.png)

聚合(message passing)之后transformer，再聚合再transformer... 有点像GAT

transformer只用在1-hop子图上



## Do Transformers Really Perform Badly for Graph Representation? (Graphormer, NIPS'21)

![img](https://pic2.zhimg.com/80/v2-a84efb638f1d904c38e7c552a6201a91_720w.webp)

#### Transformer流程

![image-20230113180020083](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230113180020083.png)

#### Centrality Encoding

![image-20230113173346001](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230113173346001.png)

其中z-是入度对应的embedding，z+是出度，均可学习

#### Spatial Encoding

![image-20230113175519840](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230113175519840.png)

我们可以将 ϕ(vi,vj) 定义为节点 vi 和 vj 之间最短路径的距离。如果不连通的话则赋给其一个特殊值-1。

#### Edge Encoding

![image-20230113175631822](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230113175631822.png)

首先找出从节点 vi 和 vj的最短路径 SPij=(e1,e2,…,eN) , 然后将路径上的特征与一个可训练的embedding逐个做点积，之后计算残差项 cij 



224*224=40000

## Pure Transformers are Powerful Graph Learners (NIPS'22)

![image-20230113162747739](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230113162747739.png)

![image-20230113163127733](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230113163127733.png)



## KG-Bert

<img src="C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230114153752522.png" alt="image-20230114153752522" style="zoom: 67%;" />



