# Nlp_任务一：基于机器学习的文本分类

## 一、实验目的
实现基于logistic/softmax regression的文本分类

## 二、实验过程
1. 环境配置：
    - 实验平台：VScode
    - GPU：NVIDIA GeForce MX450
    - 处理器：11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz  
    - 操作系统：Windows10
    - Python版本：3.10.18

2. 数据集选择：
Classify the sentiment of sentences from the Rotten Tomatoes dataset

3. 数据集划分
采用scikit-learn 库的train_test_split()函数进行数据集的划分，直接在特征提取初始化过程中调用该函数。

4. 特征提取
    - Bag-of-Word
每个句子按照单词划分，把单词统一大小写后写入字典格式词汇表。以词袋总体词汇量为基准，词袋中存在的单词记为1，不存在的记为0，不考虑词序，将原始文本数据转为0-1向量。
    - N-gram
BoW相当于1-gram，只提取文本内容，不考虑词序，N-gram在此基础上可以设定1-gram一直到N-gram的新增词汇，比如I love you用Bow提取特征，词汇表中只有I 、love、you，如果采用2维的2-gram，在此基础上会新增i_love、love_you到词汇表，从而提供词序信息，但是会增加计算量，本次任务中选择二维的2-gram。

5. 分类器
softmax regression
softmax回归是LR逻辑回归在多分类问题上的应用推广。
给定输入x，输出y向量，y的长度是提供的分类类别个数，y向量的每个内容代表对应类别的概率，所有概率和为1。
要实现softmax分类的关键在于找到合适的参数矩阵W

6. 损失函数
为了评估分类器的分类效果，选择适合分类任务的交叉熵损失函数，通过计算损失函数最小值，求解最优参数矩阵W

7. 梯度下降
采用梯度下降确定参数的更新
    1）随机梯度下降shuffle
每次随机取一个样本，直接更新
    2）整体批量下降batch
    3）小批量下降mini-batch
随机取一些样本，然后更新

## 三、实验结果
1. 实验设置
    - 样本个数：无法使用全部样本，会导致数据量过大，程序运行崩溃，选择前15000条数据
    - 学习率：0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
    - 数据集划分：7：3
    - Random_seed:2025
    - Mini-batch:10
    - Timesteps:10000

2. 实验结果
观察不同学习率下，不同梯度下降方法，以及不同特征提取方式的准确率变化。
根据下图可知，shuffle随机梯度下降和mini-batch小批量梯度下降优于batch整体批量下降，N-gram的特征提取方式优于BoW。在学习率到达1之后，shuffle随机梯度下降和mini-batch小批量梯度下降的准确率保持在较稳定水平，只有batch整体批量下降的准确率在提升。
 <img width="727" height="545" alt="figure" src="https://github.com/user-attachments/assets/da4129b6-b38e-458b-bb2c-4ca201562e1f" />

用损失函数可视化模型训练效果，固定学习率为0.1，训练次数10000次
<img width="1280" height="624" alt="Fi1" src="https://github.com/user-attachments/assets/b525aead-d00b-42f9-b9aa-6b20428354dd" />
N-gram特征提取在训练集表现更优，所有梯度下降的损失率在0.8左右，但词袋模型在测试集表现更优，损失率在1.5左右，随机梯度下降在训练后期出现上升，可能存在过拟合情况。


