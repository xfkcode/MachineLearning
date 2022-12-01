# 基于K-近邻的车牌号识别⭐
## 🎯K-近邻 ***(k-nearest neighbor, k-NN)***
- 👁️原理：  
  一个样本数据集合，也称训练样本集，样本集中每个数据都存在标签，即样本集中每一个数据与所属分类的对应关系。  
  输入新数据，比较新数据与样本集数据对应的特征，然后算法提取样本集最相似数据(最近邻)的分类标签。  
  一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。  
  最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。
- 🧠***k*** **-** **近邻**算法
  1. 计算已知类别数据集中的点与当前点之间的距离；
  2. 按照距离递增次序排序；
  3. 选取与当前点距离最小的 ***k*** 个点；
  4. 确定前 ***k*** 个点所在类别的出现频率；
  5. 返回前 ***k*** 个点所出现频率最高的类别作为当前点的预测分类。
- 📐距离度量  
  - ***Minkowski*** ( $L_{\lambda}$ )  
  $$d(i,j)=[\sum_{k=1}^{p} {(x_{ik}-x_{jk})^2}]^\frac{1}{\lambda}$$  
    - **欧几里得距离** ( $\lambda=2$ )  
  $$d_{ij}=\sqrt{\sum_{k=1}^{p} {(x_{ik}-x_{jk})^2}}$$ 
    - **曼哈顿距离** ( $\lambda=1$ )  
  $$d(i,j)=\sum_{k=1}^{p} {|(x_{k}(i)-x_{k}(j))|}$$
    - **切比雪夫距离** ( $L_{\infty}$ )
  $$d(i,j)=\underset{k}{max} {|(x_{k}(i)-x_{k}(j))|}$$ 

  📢***Bray-Curtis Dist***,***Canberra Dist***... ...
- 💾属性归一化 ***(Normalization)***  
  邻居间的距离可能被某些取值特别大的的属性所支配
  - ***Min-Max Normalization***
  $\frac{x_{i}-min(x_i)}{max(x_i)-min(x_i)}$  
  将训练集中某一列数值特征（假设是第 ***i*** 列）的值缩放到 **[** ***0,1*** **]** 之间。  
  ```python
  '''
  函数说明：数据归一化
  Parameters:
      dataSet - 原始数据
  Returns:
      normDataSet - 归一化数据
  '''
  def normdata(dataSet):
      # 获得数据的最小值
      minVals = dataSet.min(0)
      maxVals = dataSet.max(0)
      # 最大值和最小值的范围
      ranges = maxVals - minVals
      # shape(dataSet)返回dataSet的矩阵行列数
      normDataSet = np.zeros(np.shape(dataSet))
      # 返回dataSet的行数
      m = dataSet.shape[0]
      # 原始值减去最小值
      normDataSet = dataSet - np.tile(minVals, (m, 1))
      # 除以最大和最小值的差,得到归一化数据
      normDataSet = normDataSet / np.tile(ranges, (m, 1))
      return normDataSet
  ```
  - ***Z—score*** **标准化**
  $\frac{x_{i}-\bar{x}}{\sigma(x)}$   
  将训练集中某一列数值特征（假设是第 ***i*** 列）的值缩放成 **均值** ***0***，**方差** ***1*** 的状态。

  📢***Log***,***Sum***,**指数**,**正切**... ...
- 🏋️属性加权 ***(Weighted)***
  $$📌d_{WE}(i,j)=[\sum_{k=1}^{p} {W_k(x_{ik}-x_{jk})^2}]^\frac{1}{2}📌$$ 
  - $W_k=0$ $\longrightarrow$消除对应维度（特征选择）
  - 加权方法：**互信息**
    - ***entropy***
    - ***Joint entropy***
## 📡数据提取(图像)
- 数据概览
  - 🚦🏍️使用已经分割好的车牌图片作为数据集：  
    包括数字 ***0-9***、字母 ***A-Z*** （**不包含** ***O*** **和** ***I***）以及省份简称共 ***65*** 个类，编号从 ***0*** 到 ***64***。  
  - 💾🌡️数据已经分成了训练集和测试集，里面的文件夹用 ***label*** 编号命名。  
    一个文件夹下的所有图片都属于该文件夹对应的类，每个图片都是 ***(20,20)*** 的二值化灰度图。  
- **训练集数据**："./data/train/" 📂路径下 ***15954*** 张图片📸
- **测试集数据**："./data/test/" 📂路径下 ***4665***  张图片🖼️  
[***dataDownload*****链接**](https://github.com/xfkcode/MachineLearning/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E5%9F%BA%E4%BA%8EK-%E8%BF%91%E9%82%BB%E7%9A%84%E8%BD%A6%E7%89%8C%E5%8F%B7%E8%AF%86%E5%88%AB/data.zip)👈  
  
✔️⚙️借助 ***PIL*** 库将图片转化为向量：
```python
from PIL import Image
import numpy as np

img = Image.open('data/train/0/4-3.jpg')  # 打开图片
img  # 显示图片

pixels = np.array(img)  # 转化为 numpy 矩阵
pixels.shape
->(20,20)
```
## ♻️模型构建
### 🧰***sklearn*** 实现，***sklearn.neighbors.KNeighborsClassifier*** 
```python
class KNeighborsClassifier(n_neighbors: int = 5, weights: str = "uniform", algorithm: str = "auto",
                           leaf_size: int = 30, p: int = 2, metric: str = "minkowski")
```
### ♟️扩展：**自主** 实现, **K-近邻欧氏距离模型**
```python
import operator
# K-近邻欧氏距离模型
class K_NN:
    def __init__(self,Kvalue=5,distance='eucl') -> None:
        self.Kvalue=Kvalue # kNN算法参数,选择距离最小的k个点，默认5
        self.distance=distance # 距离度量方法，默认欧氏距离

    '''
    函数说明：分类器
    Parameters:
        x - 训练集features
        y - 训练集labels
        x_test - 测试集features
    Returns:
        classResult - 分类结果
    '''
    def classify(self,x,y,x_test):
        '''
        函数说明：欧氏距离
        Parameters:
            x - 训练集features
            x_test_each - 测试集features单条样本
        Returns:
            distances - 距离数组
        '''
        def Euclidean_distance(x,x_test_each):
            x_sample_num = x.shape[0] # 样本数量
            # 构造矩阵：np.tile()
            # 在列向量方向上重复 x_test_each 共 1 次(横向)
            # 行向量方向上重复 x_test_each 共 x_sample_num 次(纵向)
            diffMat = np.tile(x_test_each,(x_sample_num,1)) - x # 相减
            sqDiffMat = diffMat**2 # 平方
            # sum()所有元素相加,sum(0)列相加,sum(1)行相加
            sqDistances = sqDiffMat.sum(axis=1) # 求和
            distances = sqDistances**0.5 # 开方
            return distances
        
        '''
        函数说明: 预测,前K最小距离类别频数最大的作为分类结果
        Parameters:
            distances - 距离数组
        Returns:
            sortedClassCount[0][0] - 分类结果
        '''
        def predict(distances):
            # 返回distances中元素从小到大排序后的索引值
            sortedDistIndices = distances.argsort()
            # 定一个记录类别次数的字典
            classCount = {}
            for i in range(self.Kvalue):
                # 取出前k个元素的类别
                voteIlabel = y[sortedDistIndices[i]]
                # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
                # 计算类别次数
                classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
            # key=operator.itemgetter(1)根据字典的值进行排序
            # key=operator.itemgetter(0)根据字典的键进行排序
            # reverse降序排序字典
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            return sortedClassCount[0][0]
            
        if self.distance=='eucl':
            classResult=[]
            for x_test_each in x_test:
                distances=Euclidean_distance(x,x_test_each)
                yp=predict(distances)
                classResult.append(yp)
        elif 1:
            pass # 可扩展其他距离度量
        return classResult
```
## 🧪对比当 K 取不同值对模型效果的影响

测试 ***K-values*** 从 ***1*** 到 ***10*** 对模型效果的影响：

- 🎨画出了 ***K-values*** 对模型的影响变化  
  📉***fig1:sklearn***,📉***fig2:自主实现***  
  准确率 ***Accuracy*** : ***K*** 取 ***1*** 时准确率较高，随着 ***K*** 的增大准确率降低  
  
## 💯模型分类结果分析
👇👇根据 📉***fig1*** : ***K*** 取 ***1*** 训练 ***sklearn*** 模型
- [x] 准确率 ***Accuracy*** : ***71.6%***
- 🎭扩展分析：
  - 对比 **不同距离度量方式** 对模型效果的影响
    - **欧氏距离 : 曼哈顿距离**
  - 对比 **属性权重** 对模型效果的影响
    - **平权 : 加权**
- ⭕结论
1. ✔️***相同 K 值，[欧氏距离 ,加权] > [欧氏距离 ,平权] > [曼哈顿距离 ,平权]***  
1. ✔️***K 值增大，准确率减低***
```python
sklearn模型,[欧氏距离  ,平权,k=1],准确率 --->  0.7168274383708467
sklearn模型,[曼哈顿距离,平权,k=1],准确率 --->  0.7144694533762058
sklearn模型,[欧氏距离  ,加权,k=1],准确率 --->  0.7168274383708467
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=2],准确率 --->  0.6743837084673098
sklearn模型,[曼哈顿距离,平权,k=2],准确率 --->  0.6726688102893891
sklearn模型,[欧氏距离  ,加权,k=2],准确率 --->  0.7168274383708467
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=3],准确率 --->  0.6998928188638800
sklearn模型,[曼哈顿距离,平权,k=3],准确率 --->  0.6906752411575563
sklearn模型,[欧氏距离  ,加权,k=3],准确率 --->  0.7103965702036441
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=4],准确率 --->  0.6851018220793140
sklearn模型,[曼哈顿距离,平权,k=4],准确率 --->  0.6782422293676313
sklearn模型,[欧氏距离  ,加权,k=4],准确率 --->  0.7078242229367632
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=5],准确率 --->  0.6928188638799572
sklearn模型,[曼哈顿距离,平权,k=5],准确率 --->  0.6827438370846731
sklearn模型,[欧氏距离  ,加权,k=5],准确率 --->  0.7016077170418007
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=6],准确率 --->  0.6885316184351554
sklearn模型,[曼哈顿距离,平权,k=6],准确率 --->  0.6780278670953912
sklearn模型,[欧氏距离  ,加权,k=6],准确率 --->  0.6992497320471597
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=7],准确率 --->  0.6923901393354770
sklearn模型,[曼哈顿距离,平权,k=7],准确率 --->  0.6801714898177921
sklearn模型,[欧氏距离  ,加权,k=7],准确率 --->  0.6992497320471597
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=8],准确率 --->  0.6855305466237942
sklearn模型,[曼哈顿距离,平权,k=8],准确率 --->  0.6733118971061093
sklearn模型,[欧氏距离  ,加权,k=8],准确率 --->  0.6958199356913183
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=9],准确率 --->  0.6872454448017149
sklearn模型,[曼哈顿距离,平权,k=9],准确率 --->  0.6754555198285102
sklearn模型,[欧氏距离  ,加权,k=9],准确率 --->  0.6949624866023579
---------------------------------------------------------------
sklearn模型,[欧氏距离  ,平权,k=10],准确率 -->  0.6831725616291533
sklearn模型,[曼哈顿距离,平权,k=10],准确率 -->  0.6718113612004287
sklearn模型,[欧氏距离  ,加权,k=10],准确率 -->  0.6943193997856377
---------------------------------------------------------------
```
👇👇根据 📉***fig2*** : ***K*** 取 ***1*** 训练 **自主实现** 模型
- [x] 准确率 ***Accuracy*** : ***71.7%***  
  📢由于归一化处理数据的结果，准确率相较sklearn有微小提高❗❗❗
```python
自主实现模型,[欧氏距离  ,平权,k=1],准确率 ---> 0.7172561629153269
```
---
> ✍️ [邢福凯 (xfkcode@github)](https://github.com/xfkcode)  
> 📅 *写于 2022年10月*