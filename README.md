文件结构:
- main.py: 训练和测试代码
- utils.py: 工具函数
- stopwords.txt: 停用词表
- report.md: 项目报告
- predictions.jsonl: 经过fasttext打标后的5000条fineweb数据
- requirements.txt: 项目依赖

运行方法
1. (可选)下载[openwebmath](https://huggingface.co/datasets/open-web-math/open-web-math)和[fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)中的部分数据文件，分别存放到data/positive和data/negative中。如此，则训练时不必远程下载数据集
2. 运行main.py文件，进行训练及测试。