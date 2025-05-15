from datasets import load_dataset
import random
import fasttext

from utils import *
import json
import os

SAMPLE_SIZE=200000
PREDICT_SIZE=5000



def prepare_data(train_file, val_file, test_file, predict_file):
    #准备数据
    if not os.path.exists("data/positive") or not os.path.exists("data/negative"):
        # 下载数据集
        positive_dataset = load_dataset("open-web-math/open-web-math", split='train', streaming=True)    
        negative_dataset = load_dataset("HuggingFaceFW/fineweb",name="CC-MAIN-2024-51", split="train", streaming=True)
    else:
        # 加载本地数据集
        positive_dataset = load_dataset("data/positive",  streaming=True, split="train")
        negative_dataset = load_dataset("data/negative", streaming=True, split="train")

    print('dataset loaded')

    positive_data = []
    negative_data = []
    predict_data = []

    #取正样本
    for i, item in enumerate(positive_dataset):
        if i >= SAMPLE_SIZE:
            break
        positive_data.append(item)

    #构建数学词汇词典
    stop_words = get_stop_words()
    unigram_dict, bigram_dict = get_math_dict(positive_data, stop_words)

    #取负样本
    for i, item in enumerate(negative_dataset):
        if len(negative_data)<SAMPLE_SIZE:
            if check_likely_math(item['text'], unigram_dict,bigram_dict):
                continue
            negative_data.append(item)
        elif len(predict_data)<PREDICT_SIZE:
            predict_data.append(item)
        else:
            break
    all_data = [{'text': item['text'], 'label': '__label__positive'} for item in positive_data] + \
                [{'text': item['text'], 'label': '__label__negative'} for item in negative_data]
    random.shuffle(all_data)

    #构建训练集、验证集和测试集
    train_data = all_data[:int(0.8 * len(all_data))]
    val_data = all_data[int(0.8 * len(all_data)):int(0.9 * len(all_data))]
    test_data = all_data[int(0.9 * len(all_data)):]
    

    #保存数据
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(f"{item['label']}\t{washed(item['text'])}\n")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(f"{item['label']}\t{washed(item['text'])}\n")
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(f"{item['label']}\t{washed(item['text'])}\n")
    with open(predict_file, 'w', encoding='utf-8') as f:
        for item in predict_data:
            f.write(f"{washed(item['text'])}\n")
    print('data prepared')



def get_model(model_output_path, train_file, val_file):
    # 检查是否有已经训练好的模型，若没有则训练
    if os.path.exists(f"{model_output_path}.bin"):
        try:
            # 加载训练好的模型
            print(f"load model from {model_output_path}.bin")
            model = fasttext.load_model(f"{model_output_path}.bin")
        except Exception as e:
            #若加载失败
            print(f"load model failed, {e}")
            print("training model...")
                # 训练模型
            model = fasttext.train_supervised(
                    input=train_file,    
                    thread=8,          
                    # 使用自动调优 
                    autotuneValidationFile=val_file,
                    autotuneMetric="f1", 
                    autotuneDuration=300 
                )
            if not os.path.exists("models"):
                os.makedirs("models")
            model.save_model(f"{model_output_path}.bin")
            print(f"training completed , saved to {model_output_path}.bin")
    else:
        # 训练模型
        model = fasttext.train_supervised(
                input=train_file,    
                thread=8,          
                # 使用自动调优 
                autotuneValidationFile=val_file,
                autotuneMetric="f1", 
                autotuneDuration=300 
            )
        if not os.path.exists("models"):
            os.makedirs("models")
        model.save_model(f"{model_output_path}.bin")
        print(f"training completed , saved to {model_output_path}.bin")


def test_model(model,test_file):
    #手动测试模型，计算指标
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    test_predictions = []

    for line in test_data:
        true_label, text = line.strip().split('\t')
        labels, probs = model.predict(washed(text), k=1)
        predict_label = labels[0]
        test_predictions.append((true_label, predict_label))

    # 计算指标
    tp = sum(1 for true, pred in test_predictions if true == pred =='__label__positive')
    tn = sum(1 for true, pred in test_predictions if true == pred =='__label__negative')
    fp = sum(1 for true, pred in test_predictions if true == '__label__negative' and pred == '__label__positive')
    fn = sum(1 for true, pred in test_predictions if true == '__label__positive' and pred == '__label__negative')
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1-score: {f1_score}")
    return accuracy, precision, recall, f1_score

def annotate_data(model,predict_file):
    #标注无标签数据
    with open(predict_file, 'r', encoding='utf-8') as f:
        predict_data = f.readlines()
    print(f"Predicting labels for {len(predict_data)} examples...")

    predictions = []

    for text in predict_data:
        labels, probs = model.predict(text.strip(), k=1)
        label = labels[0]
        prob = probs[0]
        
        predictions.append({
            'text': text,
            'predicted_label': label,
            'confidence': prob
        })


    # 保存标注好的数据
    with open('predictions.jsonl', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    return predictions

if __name__ == "__main__":

    model_output_path = "models/math_filter_model"
    train_file = 'data/train.txt'
    val_file = 'data/val.txt'
    test_file = 'data/test.txt'
    predict_file = 'data/predict.txt'


    # 检查是否有已经处理好的数据，若没有则进行处理
    if not all(os.path.exists(f) for f in [train_file, val_file, test_file, predict_file]):
        prepare_data(train_file, val_file, test_file, predict_file)

    # 获取模型
    model = get_model(model_output_path, train_file, val_file)


    accuracy, precision, recall, f1_score = test_model(model,test_file)

    predictions = annotate_data(model,predict_file)



