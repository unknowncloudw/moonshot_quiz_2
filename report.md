# 题目2：FastText模型训练

## 1.数据处理

从huggingface中下载openwebmath和fineweb数据集的部分数据，用于训练。预计从openwebmath和fineweb中抽取各200000条数据，并按照8:1:1的比例划分训练集、验证集以及测试集。此外再选取5000条fineweb数据用于标注。

由于openwebmath和fineweb本身已经经过一轮清洗，数据比较干净，因此我们不需要再进行太多复杂的清洗过程。只需将所有英文字符转换为小写，去除换行符和多余空格。

注意到fineweb数据来源广泛，无法保证其中没有数学领域文本。因此在抽取负样本之前，我们先使用正样本构建一个简单的过滤器，过滤掉fineweb数据集中可能的数学领域文本。过滤方式如下：

1. 处理正样本内的所有文本，只保留英文字符，除去所有停用词 (停用词集来自 https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt)。

2. 统计正样本所有unigram和bigram出现次数，排序，选取前50个unigram和前50个bigram作为数学词表。

3. 对于某条数据，统计其内所有unigram和bigram。若至少有6个unigram出现在unigram数学词表内，并且至少有3个bigram出现在bigram数学词表内，则视为数学文本。

注意到，这里的过滤方法较为简单。参考openwebmath的清洗流程，可以使用latex符号或更复杂的模型等方式，进一步提升过滤效果。


## 2.训练

使用python fasttext包来训练模型。使用自动调优，设置自动调优时间为600秒。保存训练好的模型。

## 3.效果评估

在测试集上进行效果评估。评估表现为：

准确率: 0.975025,
精确率: 0.9731577360008042,
召回率: 0.9765447667087012,
F1-score: 0.9748483093733478。


