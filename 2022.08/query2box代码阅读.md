1. 初始化embedding时采用uniform_，设定上下界；负样本可以重复；正采样时'single'，在正确集合中挑选一个（???），负采样时'tail-batch'；warm-up-steps内开大lr；

2. 训练采用x-chain和x-inter，test时都用

3. 奇怪之处：改变dist：不用norm1，不用dist_inside

4. logsigmiod

5. test batch-size取多大

6. ```python
   parser.add_argument('--uni_weight', action='store_true', 
   ​            help='Otherwise use subsampling weighting like in word2vec')
   
   if args.uni_weight:
               positive_sample_loss = - positive_score.mean()
               negative_sample_loss = - negative_score.mean()
   else: # Default
               positive_sample_loss = - (subsampling_weight * positive_score).sum()
               negative_sample_loss = - (subsampling_weight * negative_score).sum()
               positive_sample_loss /= subsampling_weight.sum()
               negative_sample_loss /= subsampling_weight.sum()
   ```

7. 什么时候用regularization

   ```python
   regularization = args.regularization * (
                   model.entity_embedding.norm(p = 3)**3 + 
                   model.relation_embedding.norm(p = 3).norm(p = 3)**3
               )
   ```

8. 负样本个数默认128

9. np.random.choice(list)

10. ```
   >>> np.random.randint(2, size=10)
   array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
   >>> np.random.randint(5, size=(2, 4))
   array([[4, 0, 2, 1], # random
          [3, 2, 2, 0]])
   ```

11. numpy.**in1d**(*ar1*, *ar2*, *assume_unique=False*, *invert=False*)[[source\]](https://github.com/numpy/numpy/blob/v1.23.0/numpy/lib/arraysetops.py#L523-L637)

    * Returns a boolean array the same length as *ar1* that is True where an element of *ar1* is in *ar2* and False otherwise.

    * invert: True for ar1 not in ar2

    * assume_unique: if true, arrays are both assumed to be unique, which speeds up calculation. 

    * ```
      >>> test = np.array([0, 1, 2, 5, 0])
      >>> states = [0, 2]
      >>> mask = np.in1d(test, states)
      >>> mask
      array([ True, False,  True, False,  True])
      ```

12. 相同size的可以这么用：a=[0,1,2,3] ,b=[True,True,False,True]，则a[b]=[0,1,3]

13. `torch.cat([a,b],dim=i)`其实就相当于扒掉最外层(i+1)层括号concat起来，再添上(i+1)层括号

    `torch.stack`需要新建一个维度

14. `torch.index_select`

    ```python
    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-0.4664,  0.2647, -0.1228, -1.1068],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> indices = torch.tensor([0, 2])
    >>> torch.index_select(x, 0, indices)
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> torch.index_select(x, 1, indices)
    tensor([[ 0.1427, -0.5414],
            [-0.4664, -0.1228],
            [-1.1734,  0.7230]])
    ```

15. reshape会直接改变形状，view临时改变形状（相当于浅拷贝并改变形状）

    ```python
    tail = torch.index_select(self.entity_embedding, dim=0, 		     index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
    ```

16. `torch.chunk(input, chunks, dim=0)`

    ​    Attempts to split a tensor into the specified number of chunks.

17. model.train()的作用是启用 Batch Normalization 和 Dropout。

    model.train()的作用是启用 Batch Normalization 和 Dropout。

    with torch.no_grad()则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用。

18. ```python
    >>> arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> print(arr[[2,1,0],[0,1,2]])
    [7 5 3]
    >>> arr = np.zeros((3,3))
    >>> arr[[2,1,0],[0,1,2]] = 1
    >>> arr
    [[0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]]
    ```

19. `torch.argsort(input, dim=- 1, descending=False)`

    arsort()[i]表示rank=i的数的索引值

    ```python
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
            [ 0.1598,  0.0788, -0.0745, -1.2700],
            [ 1.2208,  1.0722, -0.7064,  1.2564],
            [ 0.0669, -0.2318, -0.8229, -0.9280]])
    >>> torch.argsort(a, dim=1)
    tensor([[2, 0, 3, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 3],
            [3, 2, 1, 0]])
    ```