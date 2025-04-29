from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import numpy as np
from rouge_score import rouge_scorer
import torch
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检测设备类型并配置
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS (Metal Performance Shaders) for M2 chip")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

# 1. 加载模型和分词器
checkpoint = "facebook/bart-base"  # 使用BART模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# 确保模型配置正确
model.config.use_cache = False
model.config.gradient_checkpointing = True

# 将模型移动到设备
model.to(device)

# 2. 加载 JSON 数据集
dataset = load_dataset("json", data_files={
    "train": "dialogue_dataset.json",
    "validation": "dialogue_dataset.json"
})

# 3. 分词函数
def preprocess_function(examples):
    # 为每个输入添加前缀"dialogue: "，帮助模型理解任务
    inputs = [f"dialogue: {prompt}" for prompt in examples["prompt"]]
    targets = examples["response"]
    
    # 分词输入
    model_inputs = tokenizer(
        inputs,
        max_length=128,  # BART可以处理更长的序列
        truncation=True,
        padding="max_length"
    )

    # 分词目标（response）
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. 分词数据集
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 5. 设置训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_bart",  # 更改输出目录
    per_device_train_batch_size=4,  # BART可以处理更大的batch size
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    save_steps=25,
    eval_steps=25,
    logging_steps=5,
    learning_rate=3e-5,            # BART的推荐学习率
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
    eval_strategy="steps",
    save_strategy="steps",
    predict_with_generate=True,
    fp16=False,
    gradient_accumulation_steps=8,  # 减少梯度累积步数
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_steps=100,
    lr_scheduler_type="linear",
    metric_for_best_model="loss",
    greater_is_better=False,
    max_grad_norm=1.0,
    seed=42,
    report_to="none",
    ddp_find_unused_parameters=False,
)

# 6. 创建 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# 7. 开始训练
logger.info("Starting training...")
trainer.train()

# 8. 保存模型
logger.info("Saving model...")
model.save_pretrained("./finetuned_bart")  # 更改保存目录
tokenizer.save_pretrained("./finetuned_bart")
logger.info("Training completed!")