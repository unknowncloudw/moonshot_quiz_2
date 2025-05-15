import re
import os
from copy import deepcopy
def get_stop_words():
    # 读取停用词文件
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    return stop_words

def get_math_dict(data, stop_words):
    # 构建数学词汇词典
    if os.path.exists("dict/unigram_dict.txt") and os.path.exists("dict/bigram_dict.txt"):
        with open("dict/unigram_dict.txt", "r", encoding="utf-8") as f:
            unigram_dict = f.read().splitlines()
        with open("dict/bigram_dict.txt", "r", encoding="utf-8") as f:
            bigram_dict = f.read().splitlines()
        return unigram_dict, bigram_dict
    

    unigram_appear_time = {}
    bigram_appear_time = {}
    for line in data:
        text = line['text']
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()

        for word in words:
            if word not in stop_words :
                if word not in unigram_appear_time:
                    unigram_appear_time[word] = 0
                unigram_appear_time[word] += 1
        for (w1,w2) in zip(words[:-1], words[1:]):
            if w1 not in stop_words and w2 not in stop_words:
                if (w1,w2) not in bigram_appear_time:
                    bigram_appear_time[(w1,w2)] = 0
                bigram_appear_time[(w1,w2)] += 1
    
    unigram_appear_time = sorted(unigram_appear_time.items(), key=lambda x: x[1], reverse=True)
    bigram_appear_time = sorted(bigram_appear_time.items(), key=lambda x: x[1], reverse=True)
    unigram_dict = [word for word, count in list(unigram_appear_time)[:50]]
    bigram_dict = [w1+' '+w2 for (w1, w2), count in list(bigram_appear_time)[:50]]
    if not os.path.exists("dict"):
        os.makedirs("dict")
    with open("dict/unigram_dict.txt", "w", encoding="utf-8") as f:
        for word in unigram_dict:
            f.write(word + '\n')
    with open("dict/bigram_dict.txt", "w", encoding="utf-8") as f:
        for word in bigram_dict:
            f.write(word + '\n')
    return unigram_dict, bigram_dict


def check_likely_math(text, unigram_dict, bigram_dict, uni_threshold=6,bi_threshold=3):
    # 检查文本中是否包含数学相关的单词和短语
    text_lower = text.lower()
    text_lower = re.sub(r'[^a-z\s]', '', text_lower)
    text_lower = re.sub(r'  ',' ',text_lower)
    uni_count = 0
    bi_count = 0
    for keyword in unigram_dict:
        if keyword in text_lower:
            uni_count += text_lower.count(keyword)
    for keyword in bigram_dict:
        if keyword in text_lower:
                bi_count += text_lower.count(keyword)

    return uni_count >= uni_threshold and bi_count >= bi_threshold

def washed(text):
    # 清洗文本
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

