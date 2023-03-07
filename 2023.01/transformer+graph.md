## Vision

### Swim Transformer

![image-20230120062048205](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230120062048205.png)

复杂度随图片尺寸线性增长而非平方增长

![image-20230120062145636](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230120062145636.png)

滑动窗口+patch merge使得一个patch也可以与周围的所有patch均取attention而不受window限制

![image-20230120062316434](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230120062316434.png)

不滑动窗口+滑动窗口视为一组layer，因此一般都是偶数层layer



### Heterogeneous Graph Transformer (WWW'20)

![image-20230120050447757](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230120050447757.png)

对于尾结点t，假设存在(s1, e1, t)和(s2, e2, t)，则将s1、s2分别作为K、V，t作为Q，更新t的表示H(l)[t]，有些类似于GAT，但这里是用head更新tail表示，而且对于不同的relation有不同的线性层矩阵

除此之外，还有一个依据概率采样(Mini-Batch Graph Sampling)以保持每种类型节点/关系在子图中出现概率相等，并且保证子图的密度（避免过于dense）

Discussion：真的是Transformer？



### Graph Transformer Networks (KDD'22)

![image-20230120051613597](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20230120051613597.png)

类似于上篇工作，应用在**logical query**上

将边也视为节点（给边的feature加上特殊的encoding），采用方法和上基本相同，只不过采用了Mixture-of-Experts(MoE)提高泛化能力（train时选择系数权重最大的，inference时加权加和）

首先Pre-Train(在WN18rr和FB15k-237两个数据集上)，pre-train分为随机游走阶段和固定模式阶段，接着在各个数据集上采用logical query的固定模式fine-tuning



transformer问题：

1. 难训练：不易收敛、资源开销大
2. 对于link prediction来说，只有一张图，需要特别设计任务模式
3. gnn with attention似乎对于特别dense的图也够用





### Graph Transformer Networks



dimensions

regularizer

optimizer

lr_scheduler

epochs



stopper

loss

消融实验调参吗

