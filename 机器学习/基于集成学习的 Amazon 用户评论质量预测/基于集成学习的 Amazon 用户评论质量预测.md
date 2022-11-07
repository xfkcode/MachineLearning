# 基于集成学习的 Amazon 用户评论质量预测⭐
***Amazon*** **用户评论质量**  
***reviewerID*** 用户ID 👽
***asin*** 商品ID 👔
***reviewText*** 评论 💬
***overall*** 商品评分 💯
***votes_up*** 评论点赞数 💚
***votes_all*** 评论总评价数 🗳️
***label*** 评论质量 ✔️❌

## **集成学习** ***(Ensemble Learning)***  
***'Three heads are better than one.'***
- 基本想法
  - 有时一个单个分类器表现不好，但是融合表现不错
  - 算法池中的每一个学习器都有它的权重
  - 当需要对一个新的实例作预测时
    - 每个学习器作出自己的预测
    - 然后主算法把这些结果根据权值合并起来，作出最终预测
- 集成策略
  - 平均
    - 简单平均
    - 加权平均
  - 投票
    - 多数投票法
    - 加权投票法
  - 学习
    - 加权多数
    - 堆叠 ***(Stacking)***  
      层次融合，基学习器的输出作为次学习器的输入

## **加权多数** ***(Weighted Majority)***
### ***Weighted Majority Algorithm***🧠
$a_i$ 是算法池中第 $i$ 个预测算法，每个算法对输入 $X$ 有二值输出 $\lbrace 0,1\rbrace$  
$w_i$ 对应 $a_i$ 的权值
- $\forall i,w_i \leftarrow 1$
- 对每个训练样本 $<x,c(x)>$
  - 初始化 $q_0 \leftarrow q_1 \leftarrow 0$
  - 对每个算法 $a_i$
    - **if** $a_i(x)=0$ :  
      &emsp;$q_0 \leftarrow q_0 + w_i$  
      **else** :  
      &emsp;$q_1 \leftarrow q_1 + w_i$ 
  - 如果 $q_0>q_1$ ,则预测 $c(x)=0$ ,否则预测 $c(x)=1$  
    ($q_0=q_1$ 时取任意值)
  - 对每个 $a_i\in A$
    - 如果 $a_i(x)=c(x)$ ,那么 $w_i \leftarrow \beta w_i$  
      ( $\beta \in [0,1)$惩罚系数 )  
      $\beta=0$ 时是作用在 $A$ 上的 <i><b><font color=HotPink>Halving Algorithm</font></b></i>

## ***Bagging***
- **Bagging** = **B**ootstrap **agg**rega**ting**
- Bootstrap asmpling (拔靴法/自举法采样)
  - 给定集合 $D$ ,含有 $m$ 训练样本
  - 通过从 $D$ 中均匀随机的有放回采样 $m$ 个样本构建 $D_i$
### ***Bagging Algorithm***🧠
♻️**For** $t=1,2,\ldots,T$ **Do**
1. 从 $S$ 中拔靴采样产生 $D_t$
2. 在 $D_t$ 上训练一个分类器 $H_t$

🕹️分类一个新的样本 $x\in X$ 时，通过对 $H_t$ 多数投票 [🗳️]（等权重）

## ***Boosting***
- 从失败中学习
- 基本想法
  - 给每个样本一个权值
  - $T$ 轮迭代，在每轮迭代后增大错误分类样本的权重  
    <b><font color=CornflowerBlue>更关注“难”样本</font></b>
### ***AdaBoost Algorithm***🧠
- 初始给每个样本相等权重为 $1/N$;
- ♻️**For** $t=1,2,\ldots,T$ **Do**
  1. 生成一个假设 $C_t$;
  2. 计算错误率 $\epsilon_t$:  
     $\epsilon_t$ = 所有错误分类样本权重和    
  3. $\alpha_t=\frac{1}{2}\ln{\frac{1-\epsilon_t}{\epsilon_t}}$
  4. 更新每个样本的权重：  
     正确 分类：$W_{new}=W_{old}*e^{-\alpha_t}$  
     &emsp;&emsp;**if** $\epsilon_t<0.5$ 🔽, $\epsilon_t>0.5$ 🔼  
     
     错误 分类：$W_{new}=W_{old}*e^{\alpha_t}$  
     &emsp;&emsp;**if** $\epsilon_t<0.5$ 🔼, $\epsilon_t>0.5$ 🔽  
  1. 归一化权重（权重和=1）;
- 🕹️融合所有假设 $C_t$, 各自投票权重为 $\alpha_t$ 

### ***AdaBoostM1 Algorithm***🧠
- 初始给每个样本相等权重为 $1/N$;
- ♻️**For** $t=1,2,\ldots,T$ **Do**
  1. 生成一个假设 $C_t$;
  2. 计算错误率 $\epsilon_t$:  
     $\epsilon_t$ = 所有错误分类样本权重和  
     if $\epsilon_t$ > *0.5*, 则退出循环⚠️  
  3. $\beta_t=\epsilon_t/(1-\epsilon_t)$
  4. 更新每个样本的权重：  
     正确 分类：$W_{new}=W_{old}*\beta_t$ 🔽  
     错误 分类：$W_{new}=W_{old}$ 🔼  
  5. 归一化权重（权重和=1）;
- 🕹️融合所有假设 $C_t$, 各自投票权重为 $\log{(1/\beta_t)}$