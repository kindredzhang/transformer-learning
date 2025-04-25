import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载预训练模型和分词器
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 2. 输入文本
sequence = "I've been waiting for a HuggingFace course my whole life."

# 3. 预处理：分词并转为tensors(这是针对于model的input)
inputs = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")

# 4. 模型推理：得到 logits(model的response中包含 Hidden States和 logits两个对象)
#   隐藏状态是 Transformer 模型对输入 token 序列处理后生成的向量表示
#   每个 token 对应一个高维向量（这里是 768 维），这些向量捕获了 token 的语义和上下文信息    

#   Logits是 Transformer 模型输出的未归一化的分数，用于表示模型对输入序列中每个类别的置信度
outputs = model(**inputs)
logits = outputs.logits

# 5. 后处理：将 logits 转为概率 
#   Logits虽然本身已经是一个置信度值但是通常需要讲过softmax处理后得到概率值作为最终结果
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# 6. 获取标签和概率
label_id = torch.argmax(probabilities, dim=-1).item()
label = model.config.id2label[label_id]
score = probabilities[0, label_id].item()

# 7. 输出结果
print(f"Sentence: {sequence}")
print(f"Label: {label}")
print(f"Score: {score:.4f}")
