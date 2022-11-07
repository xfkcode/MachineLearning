# 基于集成学习的 Amazon 用户评论质量预测⭐
***Amazon*** **用户评论质量**  
***reviewerID*** 用户ID 👽
***asin*** 商品ID 👔
***reviewText*** 评论 💬
***overall*** 商品评分 💯  
***votes_up*** 评论点赞数 💚
***votes_all*** 评论总评价数 🗳️
***label*** 评论质量 ✔️❌

## 📮**集成学习** ***(Ensemble Learning)***  
***'Three heads are better than one.'*** 🤔
- 基本想法💡
  - 有时一个单个分类器表现不好，但是融合表现不错
  - 算法池中的每一个学习器都有它的权重
  - 当需要对一个新的实例作预测时
    - 每个学习器作出自己的预测
    - 然后主算法把这些结果根据权值合并起来，作出最终预测
- 集成策略🎲
  - 🎰平均
    - 简单平均
    - 加权平均
  - 🗳️投票
    - 多数投票法
    - 加权投票法
  - 📗学习
    - 加权多数
    - 堆叠 ***(Stacking)***  
      🧱层次融合，基学习器的输出作为次学习器的输入

## ⚖️**加权多数** ***(Weighted Majority)***
### ***Weighted Majority Algorithm***🧠
$a_i$ 是算法池中第 $i$ 个预测算法，每个算法对输入 $X$ 有二值输出 $\lbrace 0,1\rbrace$  
$w_i$ 对应 $a_i$ 的权值
- $\forall i,w_i \leftarrow 1$
- 对每个训练样本 $[x,c(x)]$
  - 初始化 $q_0 \leftarrow q_1 \leftarrow 0$
  - 对每个算法 $a_i$
    - **if** $a_i(x)=0$ :  
      &emsp; $q_0 \leftarrow q_0 + w_i$  
      **else** :  
      &emsp; $q_1 \leftarrow q_1 + w_i$ 
  - 如果 $q_0>q_1$ ,则预测 $c(x)=0$ ,否则预测 $c(x)=1$  
    ( $q_0=q_1$ 时取任意值 )
  - 对每个 $a_i\in A$
    - 如果 $a_i(x)=c(x)$ ,那么 $w_i \leftarrow \beta w_i$  
      ( $\beta \in [0,1)$惩罚系数 )  
      $\beta=0$ 时是作用在 $A$ 上的 <i><b><font color=Gold>Halving Algorithm</font></b></i>

## 🥡***Bagging***
- **Bagging** = **B**ootstrap **agg**rega**ting**
- Bootstrap asmpling (拔靴法/自举法采样)
  - 给定集合 $D$ ,含有 $m$ 训练样本
  - 通过从 $D$ 中均匀随机的有放回采样 $m$ 个样本构建 $D_i$
### ***Bagging Algorithm***🧠
♻️**For** $t=1,2,\ldots,T$ **Do**
1. 从 $S$ 中拔靴采样产生 $D_t$
2. 在 $D_t$ 上训练一个分类器 $H_t$

🥽分类一个新的样本 $x\in X$ 时，通过对 $H_t$ 多数投票 [🗳️]（等权重）
```python
class Bagging():
    def __init__(self,baseClassifier='DT', numIt=50) -> None:
        self.baseClassifier = baseClassifier # 基分类器
        self.numIt = numIt # 循环次数（基分类器个数）
        self.estimators = [] # 存储基分类器
    
    '''
    函数说明：模型训练函数
    Parameters:
        data - 训练集特征
        labels - 训练集标签
    Returns:
        返回 numIt 个训练后的基分类器
    '''
    def fit(self,data,labels):
        m = np.shape(data)[0]
        for time in range(self.numIt):
            index = np.random.choice(m,size=(m),replace=True) # 有放回随机抽样，生成样本的随机索引序列
            datarandom = [] # 存储有放回随机抽样产生的新训练样本
            for i in index:
                datarandom.append(data[i])
            datarandom = np.array(datarandom)
            # 基分类器选择
            if self.baseClassifier == 'DT':
                clf = DecisionTreeClassifier()
            elif self.baseClassifier == 'SVM':
                clf = SVC()
            else:
                pass # 可扩展更多的基分类器
            clf.fit(data,labels)
            self.estimators.append(clf)
        return self
    
    '''
    函数说明：预测函数
    Parameters:
        data_test - 测试集特征
    Returns:
        返回预测结果
    '''   
    def predict(self,data_test):
        m =  np.shape(data_test)[0]
        predictions = np.zeros(m)
        # 融合所有基分类器预测结果，等权重投票产生最终的预测结果
        for time in range(self.numIt):
            clf = self.estimators[time]
            y_predict = clf.predict(data_test)
            predictions += np.array(y_predict)
        return [1 if i>=self.numIt/2 else 0  for i in predictions]
```
## 🥾***Boosting***
- 从失败中学习
- 基本想法
  - 给每个样本一个权值
  - $T$ 轮迭代，在每轮迭代后增大错误分类样本的权重  
    <b><font color=Gold>更关注“难”样本</font></b>
### 1️⃣***AdaBoost Algorithm***🧠
- 初始给每个样本相等权重为 $1/N$ ;
- ♻️**For** $t=1,2,\ldots,T$ **Do**
  1. 生成一个假设 $C_t$ ;
  2. 计算错误率 $\epsilon_t$ :  
     $\epsilon_t$ = 所有错误分类样本权重和    
  3. 计算 $\alpha_t$ :  
     $$\alpha_t=\frac{1}{2}\ln{\frac{1-\epsilon_t}{\epsilon_t}}$$
  4. 更新每个样本的权重：  
     <b><font color=CornflowerBlue>正确</font></b> 分类  
     **if** $\epsilon_t<0.5$ 🔽, $\epsilon_t>0.5$ 🔼
     $$W_{new} = W_{old}*e^{-\alpha_t}$$   
       
     <b><font color=HotPink>错误</font></b> 分类  
     **if** $\epsilon_t<0.5$ 🔼, $\epsilon_t>0.5$ 🔽
     $$W_{new}=W_{old}*e^{\alpha_t}$$     
       
  5. 归一化权重（权重和 =1）;
- 💣融合所有假设 $C_t$ , 各自投票权重为 $\alpha_t$ 
```python
class AdaBoost():
    def __init__(self,baseClassifier='DT',numIt=10) -> None:
        self.baseClassifier = baseClassifier # 基分类器
        self.numIt = numIt # 循环次数（基分类器个数）
        self.estimators = [] # 存储基分类器
        self.alphas = [] # 存储投票权重
    
    '''
    函数说明：模型训练函数
    Parameters:
        data - 训练集特征
        labels - 训练集标签
    Returns:
        返回训练后的基分类器以及投票权重
    '''
    def fit(self,data,labels):
        m = np.shape(data)[0]
        W = np.ones(m) / m # 样本权重，初始相等（1/样本数量）
        aggClass = np.zeros(m)
        for i in range(self.numIt):
            # 基分类器选择
            if self.baseClassifier == 'DT':
                clf = DecisionTreeClassifier()
            elif self.baseClassifier == 'SVM':
                clf = SVC()
            else:
                pass # 可扩展更多的基分类器
            clf.fit(data,labels,sample_weight=W)
            baseclass = clf.predict(data)
            error = np.sum(W * np.where(baseclass != labels, 1, 0)) # 计算加权误差
            if error==0.5:
                break
            # 根据误差更新样本权重
            alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
            if alpha==0.0: break
            self.alphas.append(alpha)
            errorArr = [1 if j else -1 for j in np.array(labels).T == baseclass]
            expon = -1 * alpha * np.array(errorArr)
            W = np.multiply(W, np.exp(expon))
            W = W / W.sum() # 归一化
            
            self.estimators.append(clf)
            self.alphas.append(alpha)
            
            #计算AdaBoost误差，当误差为0的时候，退出循环
            aggClass += alpha * baseclass                                 
            errorRate =  np.sum(np.where(np.sign(aggClass) != labels, 1, 0)) / m
            if errorRate == 0.0: 
                break
        return self
    
    '''
    函数说明：预测函数
    Parameters:
        data_test - 测试集特征
    Returns:
        返回预测结果
    '''
    def predict(self,data_test):
        m = np.shape(data_test)[0]
        aggClass = np.zeros(m)
        for time in range(len(self.estimators)):
            clf = self.estimators[time]
            y_predict = clf.predict(data_test)
            aggClass += self.alphas[time] * y_predict
        return np.sign(aggClass)
```
### 2️⃣***AdaBoostM1 Algorithm***🧠
- 初始给每个样本相等权重为 $1/N$ ;
- ♻️**For** $t=1,2,\ldots,T$ **Do**
  1. 生成一个假设 $C_t$ ;
  2. 计算错误率 $\epsilon_t$ :  
     $\epsilon_t$ = 所有错误分类样本权重和  
     if $\epsilon_t$ > *0.5*, 则退出循环⚠️  
  3. 计算 $\beta_t$
     $$\beta_t=\epsilon_t/(1-\epsilon_t)$$
  4. 更新每个样本的权重：  
     <b><font color=CornflowerBlue>正确</font></b> 分类 🔽 
     $$W_{new}=W_{old}*\beta_t$$
     <b><font color=HotPink>错误</font></b> 分类 🔼
     $$W_{new}=W_{old}$$    
  5. 归一化权重（权重和 =1）;
- 💣融合所有假设 $C_t$ , 各自投票权重为 $\log{(1/\beta_t)}$
```python
class AdaBoostM1():
    def __init__(self,baseClassifier='DT',numIt=10) -> None:
        self.baseClassifier = baseClassifier # 基分类器
        self.numIt = numIt # 循环次数（基分类器个数）
        self.estimators = [] # 存储基分类器
        self.betas = [] # 存储投票权重
    
    '''
    函数说明：模型训练函数
    Parameters:
        data - 训练集特征
        labels - 训练集标签
    Returns:
        返回训练后的基分类器以及投票权重
    '''
    def fit(self,data,labels):
        m = np.shape(data)[0]
        W = np.ones(m) / m
        aggClass = np.zeros(m)
        for i in range(self.numIt):
            # 基分类器选择
            if self.baseClassifier == 'DT':
                clf = DecisionTreeClassifier()
            elif self.baseClassifier == 'SVM':
                clf = SVC()
            else:
                pass # 可扩展更多的基分类器
            clf.fit(data,labels,sample_weight=W)
            baseclass = clf.predict(data)
            error = np.dot(W.T, baseclass != labels) # 计算加权误差
            # 如果误差大于0.5退出循环
            if error > 0.5:
                break
            self.estimators.append(clf)
            # 根据误差更新样本权重
            beta = float( max(error, 1e-16) / (1.0 - error))
            self.betas.append(beta)
            update = np.array([beta if j else 1 for j in np.array(labels).T == baseclass])
            W = np.multiply(W,update)
            W = W / W.sum() # 归一化
           
            #计算AdaBoostM1误差，当误差为0的时候，退出循环
            aggClass += np.log(1/beta) * baseclass                                 
            errorRate =  np.sum(np.where(np.sign(aggClass) != labels, 1, 0)) / m
            if errorRate == 0.0: 
                break
        return self
    
    '''
    函数说明：预测函数
    Parameters:
        data_test - 测试集特征
    Returns:
        返回预测结果
    '''
    def predict(self,data_test):
        m = np.shape(data_test)[0]
        aggClass = np.zeros(m)
        for time in range(len(self.estimators)):
            clf = self.estimators[time]
            y_predict = clf.predict(data_test)
            aggClass += np.log(1/self.betas[time]) * y_predict
        return np.sign(aggClass)
```
## 📡数据概览
#### 🔰导入数据
- **训练集**："./data/" 📂路径下 *train.csv* 📝文件  
  包含总共 ***57039*** 条数据样本💾  
  每条样本包含 ***7*** 个特征🐞：  
  - *reviewerID，asin，reviewText，overall，votes_up，votes_all，label*
- **测试集**："./data/" 📂路径下 *test.csv* 📝文件  
  包含总共 ***11208*** 条数据样本💾  
  每条样本包含 ***5*** 个特征🐞：  
  - *Id，reviewerID，asin，reviewText，overall*  
  
  **测试集标签**："./data/" 📂路径下 *groundTruth.csv* 📝文件  
  - *Id，Expected*  
[***dataDownload*****链接**](https://github.com/xfkcode/MachineLearning/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E5%9F%BA%E4%BA%8E%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E7%9A%84%20Amazon%20%E7%94%A8%E6%88%B7%E8%AF%84%E8%AE%BA%E8%B4%A8%E9%87%8F%E9%A2%84%E6%B5%8B/data.zip)👈

[📢]：测试集标签分离文件存储，*Id* 与测试集特征样本一一对应，*Expected* 即 *label*。
```python
# 读入csv文件为pandas的DataFrame
train_df = pd.read_csv('./data/train.csv', sep='\t')
test_df = pd.read_csv('./data/test.csv',sep='\t')
testlabels_df = pd.read_csv('./data/groundTruth.csv')
df.head(3) # 显示前三条数据
df.info() # 显示数据基本信息,可检查是否有数据丢失
df.describe() # 显示数据统计摘要
```
## 🧪实验方案 + 特征工程
🧰***sklearn*** 文本向量化工具  
- [***sklearn.feature_extraction.text.CountVectorizer***](https://github.com/xfkcode/MachineLearning/blob/main/python%E5%B7%A5%E5%85%B7/sklearn/sklearn%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B01.ipynb)  
  将文本文档集合转换为计数矩阵。
- [***sklearn.feature_extraction.text.TfidfVectorizer***](https://github.com/xfkcode/MachineLearning/blob/main/python%E5%B7%A5%E5%85%B7/sklearn/sklearn%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B02.ipynb)  
  将文档集合转换为TF-IDF功能矩阵。
### ***Test-1***⚗️
数据🧫
- 使用训练集前 ***2000*** 条数据作为 *Test-1* 全部数据进行数据划分测试  

⚠️电脑跑不动大规模数据❗❗❗

[📢]：训练集和测试集共同构建词表实现文本向量化

1. ***reviewText*** 评论文本向量化  
   - **全部数据** 构建词表，实现文本向量化
2. 数据划分✂️   
   训练集 : 测试集 ***(8:2)***

### ***Test-2***⚗️
数据🧫 
- 使用训练集前 ***2000*** 条数据作为 *Test-2* 训练集数据  
- 使用测试集前 ***200*** 条数据作为 *Test-2* 测试集数据  
 
⚠️电脑跑不动大规模数据❗❗❗

[📢]：仅使用训练集构建词表实现文本向量化

1. ***reviewText*** 评论文本向量化  
   - **训练集数据** 构建词表，实现文本向量化
2. 构建训练集、测试集数据

## 🕹️模型构建
* ***Bagging + SVM***
* ***Bagging + 决策树***
* ***AdaBoost + SVM***
* ***AdaBoost + 决策树***
* ***AdaBoost.M1 + SVM***
* ***AdaBoost.M1 + 决策树***
### 🧰***sklearn*** 实现
- 基分类器SVM： `SVC(C=200,kernel='rbf')`
- 基分类器DT：  &emsp;`DecisionTreeClassifier(max_depth=3)`
  
```python
Bagging_svm = BaggingClassifier(SVC(C=200,kernel='rbf'), n_estimators = 50)
Bagging_Dt = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators = 50)
adBt_svm = AdaBoostClassifier(SVC(C=200,kernel='rbf'), algorithm='SAMME', n_estimators = 10)
adBt_DT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm='SAMME', n_estimators = 10)
```
### ♟️**自主** 实现
- 基分类器SVM： `SVC()`
- 基分类器DT：  &emsp;`DecisionTreeClassifier()`

[🔗***Bagging***](##🥡***Bagging***)👈 
[🔗***Boosting***](##🥾***Boosting***)👈
```python
clfbagging=Bagging().fit(x_train,y_train)
clfAdaBoost=AdaBoost().fit(x_train,y_train)
clfAdaBoostM1=AdaBoostM1().fit(x_train,y_train)
```
## 💯对比 ***Accuracy*** 模型性能
统计对比模型准确率  
*Test-1* 🔬
- [x] ***Sklearn-models***
- [x] ***Self-models***
```python
Sklearn-models >>>
Bagging + SVM Accuracy : 0.795
Bagging + DT Accuracy : 0.76
AdaBoost + SVM Accuracy : 0.775
AdaBoost + DT Accuracy : 0.755
Self-models >>>
Bagging + DT Accuracy : 0.695
Bagging + SVM Accuracy : 0.77
AdaBoost + DT Accuracy : 0.665
AdaBoost + SVM Accuracy : 0.77
AdaBoostM1 + DT Accuracy : 0.675
AdaBoostM1 + SVM Accuracy : 0.77
```
## 📈对比 ***ROC/AUC*** 模型性能
📐画出了模型 ***ROC*** 曲线 [🔗](https://github.com/xfkcode/MachineLearning/blob/main/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E5%9F%BA%E4%BA%8E%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E7%9A%84%20Amazon%20%E7%94%A8%E6%88%B7%E8%AF%84%E8%AE%BA%E8%B4%A8%E9%87%8F%E9%A2%84%E6%B5%8B/%E5%9F%BA%E4%BA%8E%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E7%9A%84%20Amazon%20%E7%94%A8%E6%88%B7%E8%AF%84%E8%AE%BA%E8%B4%A8%E9%87%8F%E9%A2%84%E6%B5%8B.ipynb)  
💻计算了 ***AUC*** 指标  
*Test-1* 🔬
- ***sklearn-models***  
  ***sklearn*** 模型使用 `proba` 比 `predict` 效果好  
  ***AUC*** 指标均超过 ***0.6***
- ***self-models***  
自主实现模型效果不好  
👉**原因**：
*AdaBoost/AdaBoostM1* 算法每次迭代后基分类器错误率会升高，  
并保持在 ***0.5*** 左右
会使得权重更新无法正常运行，基分类器样本加权训练结果出现偏差。  

⭕目前还没有找出解决办法，后续会继续探索更新🤯

---
> ✍️ [邢福凯 (xfkcode@github)](https://github.com/xfkcode)  
> 📅 *写于 2022年11月*