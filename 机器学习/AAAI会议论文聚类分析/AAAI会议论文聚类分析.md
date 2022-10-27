# AAAI会议论文聚类分析⭐
## 🗃️聚类 ***(Unsupervised Learning)***
&emsp;将相似的对象归入同一个“类” ***cluster*** **(簇)**  
&emsp;使得同一个类中的对象互相之间关联更强  
  - [x] 同一个类中的对象相似  
  **簇/类内** *(intra-cluster)* **相似度大，距离小**
  - [x] 不同类中的对象有明显差异  
  **簇/类间** *(inter-cluster)* **相似度小，距离大**

#### 🧩聚类的类型🧩  
- 软聚类 & 硬聚类
  - 软：同一个对象可以属于不同类
  - 硬：同一个对象只能属于一个类
- 层次聚类 & 非层次聚类
  - 层次：*tree* 结构
    - 凝聚式层次聚类 (Agglomerative)  
      bottom-up 自底向上合并
    - 分列式层次聚类 (Divisive)  
      top-down 自顶向下分裂
  - 非层次： 只有一层 *flat*  
    - ***K-means*** 聚类
    - ***K-medoids*** 聚类
## 🕹️***K-means***
- ***Algorithm***🧠  
  给定一个类分配方案 $C$ ，确定每个类的均值向量： $\lbrace g_1,g_2,\ldots,g_K \rbrace$  
  给定 $K$ 个均值向量的集合 $\lbrace g_1,g_2,\ldots,g_K \rbrace$ ，把每个对象分配给距离均值最近的类  
  重复上述过程直到评价函数值不发生变化
   1. 初始化
      随机选择 $K$ 个种子，使得 $d(g_i,g_j)>d_{min}$     
   2. 类的分配  
      根据最近距离把点分配给各类  
      $$Cluster(u_i)=\underset{g_i\in\lbrace g_1,g_2,\ldots,g_K\rbrace}{argmin}(d(u_i,g_i))$$
   3. 更新中心  
      计算新的类中心
      $$g_j=\frac{1}{n}\sum_{u_i\in j^{th}cluster}{u_i}$$
        ***Repeat 2&3***
## 🕹️***K-medoids***
用 <b><font color=HotPink>最靠近类中心的对象</font></b> 作为类的参考点  
而不是类的均值 ***(K-means)***
- ***Algorithm***🧠  
  找到 $n$ 对象中的 $k$ 个类，随机确定每个类的代表对象  
  迭代：  
  - 其他所有对象根据距离最近的类中心进行类的分配
  - <b><font color=CornflowerBlue>计算使得 *cost* 最小的类中心</font></b>
  
  重复直到不再发生变化  
  代价函数：类内对象与类中心的平均不相似度

- ⏫改进算法 ***PAM***  
  ***Partitioning Around Medoids***  
  找到 $n$ 对象中的 $k$ 个类，随机确定每个类的代表对象  
  迭代：  
  - 其他所有对象根据距离最近的类中心进行类的分配
  - <b><font color=CornflowerBlue>随机用一个非中心对象替换类中心</font></b>
  - <b><font color=CornflowerBlue>类的质量提高则保留替换</font></b>
  
  重复直到不再发生变化  
  代价函数：类内对象与类中心的平均不相似度






---
> ✍️ [邢福凯 (xfkcode@github)](https://github.com/xfkcode)  
> 📅 *写于 2022年10月*