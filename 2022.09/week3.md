## YOU CAN TEACH AN OLD DOG NEW TRICKS! ON TRAINING KNOWLEDGE GRAPH EMBEDDINGS  (ICLR'20)

旧瓶装新酒，古早的KGE模型加上好的超参也能接近甚至超过最新模型（均指不包括其他信息的朴素KGE）

* 最佳超参配置与模型和数据集有关，且彼此有很大区别：不能通过小搜索空间的网格搜索评判KGE的好坏
* 在取值为数值的hp间进行贝叶斯优化，性能提高不大
* 特定超参的影响不太好直观解释，大多数超参数不很敏感，但loss是非常敏感的超参数
* loss选取中CE似乎是效果最好的

提醒：不同数据集和模型应该从一个大的搜索空间自适应地寻找最优区域，不应该人为划定



## FEW-SHOT BAYESIAN OPTIMIZATION WITH DEEP KERNEL SURROGATES  (ICLR'21)

Motivation: 将迁移学习中的HPO视作小样本学习任务，通过小样本学习学习surrogate model的初始参数，然后对目标任务进行微调

Methods：T个source tasks共用一个Surrogate，之后target task进行少量fine-tune，其中surrogate model用的是要学习参数的deep kernel

![image-20220917113110329](pic/FSBO1.png)

Loss：边缘似然<img src="pic/FSBO3.png" style="zoom:60%;" />

假设：在许多source tasks上训出的model，只要target task的分布与之相像，则直接取mean可以很快收敛<img src="pic/FSBO2.png" style="zoom:60%;" />

细节：不同source task的y全部scale到[0,1]导致normalization也必须应用到target上，于是我们对所有的source task的y取min和max得到$y_{min},y_{max}$，对于每一个batch取![image-20220917114855553](C:\Users\Xsu1023\AppData\Roaming\Typora\typora-user-images\image-20220917114855553.png)

目的是让model能够学到变化的offset/variance下的不变特征，test时通过如下方式计算：<img src="pic/FSBO4.png" alt="image-20220917121725174" style="zoom:67%;" />

fine-tune时选择source tasks里使得loss最小的HP，同时结合进化算法——该种方法对性能提升有限，但似乎能够探索到搜索空间里接近最优的区域



## TransBO: Hyperparameter Optimization via Two-Phase Transfer Learning  (KDD'22)

利用**迁移学习**方法的两阶段HPO

![](pic/TransBO1.png)

$1、M^S=agg(\{M^1,\cdots,M^K\};w)=\sum w_iM^i，其中w是可学习的模型间权重；$$2、M^{TL}=agg(\{M^S,M^T\};p)，其中p是可学习权重$

Loss采用一种可导的ranking loss：<img src="pic/TransBO2.png" alt="image-20220917173830985" style="zoom:80%;" />

整个流程：<img src="pic/TransBO3.png" style="zoom:80%;" />

## Transfer Learning based Search Space Design for Hyperparameter Tuning  (KDD'22)

首先BO方法获得promising regions: 

1) 对比source surrogate models相似度（其中$D^T$是target dataset，$\otimes$表示异或）<img src="pic\untitled1.png" style="zoom:80%;" />

2) 根据source tasks自适应选取搜索空间
   * 搜索空间$\mathcal{X}^i=\{x_j|x_j\in \mathcal{X},y^i_j\leq y_+^i\}$，其中$y^i_+$是第i个source task的$\alpha_i$分位数，该$\alpha_i$可通过如下公式调整：<img src="pic\untitled2.png" style="zoom:80%;" />，$\alpha_{min}和\alpha_{max}$是预先指定的超参，能够看出相似性越小则$\alpha$越大，因此搜索空间越大

其次生成搜索空间：

* target search space：<img src="pic\untitled3.png" style="zoom:80%;" />

* 除此之外还有投票机制，对于所有的source task利用相似度打分，然后softmax，服从该分布选出k个source task，利用此模型打分，看promising二分类是否超过半数

综合<img src="pic\untitled4.png" style="zoom:80%;" />

启发：transfer learning - 能否将一张子图视为一个source task
