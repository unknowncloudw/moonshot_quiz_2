from datasets import load_dataset
import random
import fasttext

from utils import *
import json
import os

if not os.path.exists("data"):
    positive_dataset = load_dataset("open-web-math/open-web-math", split='train', streaming=True)    
    negative_dataset = load_dataset("HuggingFaceFW/fineweb",name="CC-MAIN-2024-51", split="train", streaming=True)
else:
    positive_dataset = load_dataset("data/positive",  streaming=True, split="train")
    negative_dataset = load_dataset("data/negative", streaming=True, split="train")

sample_size = 200000
predict_size = 5000
positive_data = []
negative_data = []
predict_data = []
for i, item in enumerate(positive_dataset):
    if i >= sample_size:
        break
    positive_data.append(item)

stop_words = get_stop_words()
unigram_dict, bigram_dict = get_math_dict(positive_data, stop_words)




for i, item in enumerate(negative_dataset):
    if len(negative_data)<sample_size:
        if check_likely_math(item['text'], unigram_dict,bigram_dict):
            continue
        negative_data.append(item)
    elif len(predict_data)<predict_size:
        predict_data.append(item)

all_data = [{'text': item['text'].lower(), 'label': '__label__positive'} for item in positive_data] + \
              [{'text': item['text'].lower(), 'label': '__label__negative'} for item in negative_data]
random.shuffle(all_data)
train_data = all_data[:int(0.8 * len(all_data))]
val_data = all_data[int(0.8 * len(all_data)):int(0.9 * len(all_data))]
test_data = all_data[int(0.9 * len(all_data)):]

with open('train.txt', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(f"{item['label']} {item['text']}\n")
with open('val.txt', 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(f"{item['label']} {item['text']}\n")
with open('test.txt', 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(f"{item['label']} {item['text']}\n")


model_output_path = "math_filter_model"
train_file = 'train.txt'
val_file = 'val.txt'
model = fasttext.train_supervised(
        input=train_file,
        lr=0.1,            
        epoch=25,          
        wordNgrams=2,      
        dim=300,           
        loss='softmax',    
        bucket=2000000,    
        thread=4,          
        # 使用自动调优 
        autotuneValidationFile=val_file,
        autotuneMetric="f1", 
        autotuneDuration=600 
    )
model.save_model(f"{model_output_path}.bin")
print(f"模型训练完成，已保存到: {model_output_path}.bin")


result = model.test(val_file)
result_per_label = model.test_label(val_file)

print(f"验证集样本数: {result[0]}")
print(f"整体精确率 (P@1): {result[1]:.4f}") # Precision at 1
print(f"整体召回率 (R@1): {result[2]:.4f}") # Recall at 1

print("\n各标签性能:")
for label, metrics in result_per_label.items():
    print(f"  标签: {label}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-score:  {metrics['f1score']:.4f}")


    # Predict labels for the unlabeled predict_data
    print(f"\nPredicting labels for {len(predict_data)} examples...")

    positive_count = 0
    predictions = []

    for item in predict_data:
        text = item['text'].lower()
        # Get prediction and probability
        labels, probs = model.predict(text, k=1)
        label = labels[0]
        prob = probs[0]
        
        # Track positive predictions
        if label == '__label__positive':
            positive_count += 1
        
        predictions.append({
            'text': text,
            'predicted_label': label,
            'confidence': prob
        })


    # Save predictions to a file
    with open('predictions.jsonl', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    