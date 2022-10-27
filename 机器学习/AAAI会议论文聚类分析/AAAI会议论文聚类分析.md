# AAAI会议论文聚类分析⭐
***AAAI*** **会议论文**  
***Title*** 题目📌
***Authors*** 作者✍️
***Groups*** 组别🗄️
***Keywords*** 关键词🗝️
***Topics*** 主题🏷️
***Abstract*** 摘要📑  
## 🗃️聚类 ***(Unsupervised Learning)***
将相似的对象归入同一个“类” ***cluster*** **(簇)**  ，使得同一个类中的对象互相之间关联更强  
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
    - 凝聚式层次聚类 ***(Agglomerative)***  
      bottom-up 自底向上合并
    - 分列式层次聚类 ***(Divisive)***  
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
## 📡数据概览
#### 🔰导入数据
- **数据集**："./data/" 📂路径下 *[UCI] AAAI-14 Accepted Papers - Papers.csv* 📝文件  
  包含总共 ***398*** 条数据样本💾  
  每条样本包含 ***6*** 个特征🌵：  
  - *title，authors，groups，keywords，topics，abstract*
```python
data_df = pd.read_csv('./data/[UCI] AAAI-14 Accepted Papers - Papers.csv') # 读入csv文件为pandas的DataFrame
data_df.head(3) # 显示前三条数据
data_df.info() # 显示数据基本信息
data_df.describe() # 显示数据统计摘要
```
#### 🔬数据预处理
- 🥽检查数据是否有丢失 ***(NaN)***
  - ⚠️***Groups***
    - *Index*: ***211***,***340***❗❗❗
  - ⚠️***Topics***
    - *Index*: ***344***,***364***,***365***,***388***❗❗❗

📣 *groups* 存在 ***2*** 条样本数据丢失  
📣 *topics* 存在 ***4*** 条样本数据丢失  

✔️🛠️**将丢失数据替换为 `' '` 空格字符**
```python
# 检查数据是否有丢失(NaN)
fts=data_df.columns.values.tolist()
for ft in fts:
    for i,item in enumerate(data_df[ft]):
        if pd.isnull(item):
            print('数据丢失(NaN)\n>>>feature:%s,index:%d'%(ft,i))
            data_df[ft][i]=' ' # 将丢失的数据替换为 ' ' 空格字符
```
- 📦整合数据  
  合并特征，整合为完整的 *paper* 基本信息

✔️⚙️借助 ***pandas*** *DataFrame* 列合并(+)，创建新的列 `paper`  
⭕注：列合并操作(str+str)连接没有空格，需要额外添加空格进行分词
```python
# 合并特征,整合为完整的paper信息
data_df['sp']=[' ' for x in range(data_df.shape[0])] # 合并需要+空格
data_df['paper']=data_df['title']+data_df['sp']\
                +data_df['authors']+data_df['sp']\
                +data_df['groups']+data_df['sp']\
                +data_df['keywords']+data_df['sp']\
                +data_df['topics']+data_df['sp']\
                +data_df['abstract']+data_df['sp']
data_df.paper
>>>
Name: paper, Length: 398, dtype: object
```
## 📜文本向量化
### 🧰***sklearn*** 文本向量化工具  
- [***sklearn.feature_extraction.text.CountVectorizer***](https://github.com/xfkcode/MachineLearning/blob/main/python%E5%B7%A5%E5%85%B7/sklearn/sklearn%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B01.ipynb)  
  将文本文档集合转换为计数矩阵。
- [***sklearn.feature_extraction.text.TfidfVectorizer***](https://github.com/xfkcode/MachineLearning/blob/main/python%E5%B7%A5%E5%85%B7/sklearn/sklearn%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B02.ipynb)  
  将文档集合转换为TF-IDF功能矩阵。  
  
  🎪***TF-TDF***  
  (*Term Frequency-Inverse Document Frequency*, 词频-逆文本频率)  
  一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章  
  $📌TF-IDF(x)=TF(x)*IDF(x)📌$
  - ***TF***  
  $TF(x)=\frac{\text{Number of the term appears in the doc}}{\text{Total number of words in the doc}}$
  - ***IDF***  
  $IDF(x)=\log{\frac{N}{N(x)}}$  
  $N$ 代表语料库中文本的总数，而 $N(x)$ 代表语料库中包含词xx的文本总数
  - **平滑** ***IDF***  
  $IDF(x)=\log{\frac{N+1}{N(x)+1}}+1$
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# max_df=0.9表示过滤一些在90%的文档中都出现过的词
# min_df=10表示在所有文档中出现低于10次的词
vectorizer1 = CountVectorizer(max_df=0.9, min_df=10)
X1 = vectorizer1.fit_transform(data_df.paper)

vectorizer2 = TfidfVectorizer(max_df=0.9, min_df=10)
X2 = vectorizer2.fit_transform(data_df.paper)
```
## ♟️模型构建
### 🧰***sklearn***实现，***sklearn.cluster.KMeans***  
```python
class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, 
tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
```
## 👀可视化分析🔭
⚠️**提示**：
1. 降维与聚类是两件不同的事情，聚类实际上在降维前的高维向量和降维后的低维向量上都可以进行，结果也可能截然不同。
2. 高维向量做聚类，降维可视化后若有同一类的点不在一起，是正常的。在高维空间中它们可能是在一起的，降维后损失了一些信息。
### 🧰***sklearn*** 数据降维工具  
- [***sklearn.decomposition.PCA***](https://github.com/xfkcode/MachineLearning/blob/main/python%E5%B7%A5%E5%85%B7/sklearn/sklearn%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B03.ipynb)  
  主成分分析法（PCA）  
  
  🎪***PCA***  
  将关系紧密的变量变成尽可能少的新变量，使这些新变量是两两不相关的，  
  即用较少的综合指标分别代表存在于各个变量中的各类信息，达到数据降维的效果。  
  - 🎯映射：将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，  
  是在原有n维特征的基础上重新构造出来的k维特征。我们要选择的就是让映射后样本间距最大的轴。  
  - 🛵过程：
    1. 样本归0
    2. 找到样本点映射后方差最大的单位向量 $\omega$  
        求 $\omega$ ,使得 $\displaystyle Var(X_{project})=\frac{1}{m} \sum_{i=1}^{m}{(X^{(i)}\cdot\omega)^2}$ 最大
```python
from sklearn.decomposition import PCA
# 降维至2维
pca = PCA(n_components=2)
X1_pca2 = pca.fit_transform(X1.toarray())
# 降维至3维
Pca = PCA(n_components=3)
X1_pca3=Pca.fit_transform(X1.toarray())
```
### 🧰***matplotlib*** 画图工具
#### 🐞***Clusters=5>>>13***
&emsp; ***K-meanns Algorithm*** 实现聚类，**聚类在降维之前**  
&emsp; **对比不同簇数聚类结果**🧪  
  

- 🌠**降维>>>2维**  
✔️⚙️借助 ***scatter2D*** 可视化聚类结果   
将数据降维至两维，由于损失了一些信息，越多 *clusters* 会观测到越多偏离的点，总体可以明显观测出 ***4-5*** 个类别

- 🌠**降维>>>3维**  
✔️⚙️借助 ***scatter3D*** 可视化聚类结果  
将数据降维至三维，同样由于信息的损失，也存在离群点，  
但相较 ***scatter2D*** 有了更强的可视性，类别之间的空间关系更加明显，聚类效果也更加直观，总体可以观测出 ***5-7*** 个类别

---
> ✍️ [邢福凯 (xfkcode@github)](https://github.com/xfkcode)  
> 📅 *写于 2022年10月*