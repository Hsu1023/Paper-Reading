1. hard_ans是什么

2. 1 vs k（答案很不一样？）所有正确答案的ranking取sum得hitsn & mrr 是否不可取？取mean得hits1m，为什么不是有一个就算？ Acc Recall(推荐系统)

3. test batch-size取多大

4. warm-up (lr-warming up)

5. norm1好处？ dist-inside

6. ```python
   self.embedding_range = nn.Parameter(
               torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
               requires_grad=False # epsilon = 2
           )
   self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
           nn.init.uniform_(
               tensor=self.entity_embedding, 
               a=-self.embedding_range.item(), 
               b=self.embedding_range.item()
           )
   ```

7. ```python
   subsampling_weight = self.count[(head, relations)] # 让entity出险频率低也能学得较好
   subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight])) #1*1 tensor
   
   def count_frequency(triples, true_tail, start=4):
           count = {}
           for triple, qtype in triples: # triple is like (4640, (20, 326, 167), 0)
               head, relations, tail = triple
               assert (head, relations) not in count
               count[(head, relations)] = start + len(true_tail[((head, relations),)])
           return count
   ###########
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

8. margin-based怎么采负样本？loss的logsigmoid和relu哪个效果好？