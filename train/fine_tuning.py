import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

# 加载模型和分词器
checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# 加载 JSON 数据集
dataset = load_dataset("json", data_files="dataset.json")

# 查看数据集的结构
print("\n数据集结构:")
print(dataset)
print("\n数据集示例:")
print(dataset["train"][0])

# 分词函数 - 简化版本
def preprocess_function(examples):
    inputs = examples["instruction"]
    targets = examples["response"]

    # 分词输入
    model_inputs = tokenizer(inputs, padding="max_length", max_length=128, truncation=True)

    # 分词目标
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    # 将填充标记替换为-100，以在计算损失时忽略它们
    for i in range(len(model_inputs["labels"])):
        model_inputs["labels"][i] = [
            -100 if token == tokenizer.pad_token_id else token
            for token in model_inputs["labels"][i]
        ]

    return model_inputs

# 处理数据集
print("\n处理数据集...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_bart",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=False,  # 如果您的GPU支持，可以设置为True
    push_to_hub=False,
)

# 创建训练器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# 训练
print("\n开始训练...")
trainer.train()

# 保存模型
print("\n保存模型...")
model.save_pretrained("./finetuned_bart")
tokenizer.save_pretrained("./finetuned_bart")
print("\n完成！")

# 测试模型
print("\n测试模型:")
test_text = "Translate this to Chinese: 'Hello, world!'"
inputs = tokenizer(test_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n输入: {test_text}")
print(f"输出: {translation}")