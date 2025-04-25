

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from datasets import load_dataset
import evaluate
import numpy as np

# 1. 从Hub加载数据集
print("正在加载MRPC数据集...")
raw_datasets = load_dataset("glue", "mrpc")
print(f"数据集加载完成，包含以下拆分: {raw_datasets.keys()}")
print(f"训练集样本数: {len(raw_datasets['train'])}")
print(f"验证集样本数: {len(raw_datasets['validation'])}")
print(f"测试集样本数: {len(raw_datasets['test'])}")

# 查看数据集的一个样本
print("\n数据集样本示例:")
print(raw_datasets["train"][0])
print("\n标签含义:")
print(raw_datasets["train"].features["label"].names)

# 2. 数据预处理
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 定义分词函数
def tokenize_function(examples):
    # 对句子对进行分词
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        # 不在这里进行padding，而是在批处理时动态padding
    )

# 对整个数据集应用分词处理
print("\n正在对数据集进行分词处理...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("分词处理完成!")

# 查看处理后的数据集结构
print(f"\n处理后的数据集特征: {tokenized_datasets['train'].column_names}")

# 移除不需要的列，保留模型需要的输入
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
# 重命名"label"列为"labels"(Trainer API要求)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 设置数据集格式为PyTorch
tokenized_datasets.set_format("torch")
#这里的tokenized_datasets对象才是真正需要训练使用到的内容，这个对象内部包含的其实是英文真实文字的 ids 表达，为了确保模型可以读得懂
print(f"\n最终处理后的数据集特征: {tokenized_datasets['train'].column_names}")

# 3. 动态填充
# 创建数据收集器，用于批处理时的动态填充
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 4. 定义评估指标
def compute_metrics(eval_preds):
    # 1. 加载MRPC任务专用的评估器
    # MRPC (Microsoft Research Paraphrase Corpus) 用于评估句子对相似度任务
    metric = evaluate.load("glue", "mrpc")
    
    # 2. 解包预测结果
    logits, labels = eval_preds  
    # 示例数据结构：
    # logits = [
    #    [-1.2, 2.3],   # 第1个样本：[不相似的分数, 相似的分数]
    #    [1.5, -0.8],   # 第2个样本
    #    [-0.5, 1.9],   # 第3个样本
    #    [-2.1, 3.4]    # 第4个样本
    # ]
    # labels = [1, 0, 1, 1]  # 真实标签：1=相似，0=不相似
    
    # 3. 将logits转换为具体预测类别
    predictions = np.argmax(logits, axis=-1)  
    # axis=-1表示在最后一个维度上取最大值的索引
    # 转换过程：
    # [-1.2, 2.3]  -> 1  (2.3分数更高，预测"相似")
    # [1.5, -0.8]  -> 0  (1.5分数更高，预测"不相似")
    # [-0.5, 1.9]  -> 1  (1.9分数更高，预测"相似")
    # [-2.1, 3.4]  -> 1  (3.4分数更高，预测"相似")
    # 得到：predictions = [1, 0, 1, 1]
    
    # 4. 使用评估器计算性能指标
    return metric.compute(predictions=predictions, references=labels)
    # 返回值示例：
    # {
    #    'accuracy': 0.8406,  # 准确率：84.06%的预测正确
    #    'f1': 0.8892        # F1分数：准确率和召回率的调和平均
    # }
    # - accuracy衡量总体预测正确的比例
    # - f1特别适合评估分类任务，尤其是在类别不平衡的情况下

# 5. 初始化模型
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
)

# 6. 定义训练参数
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch", # eval_strategy是评估策略的设置，它决定了在训练过程中何时进行模型评估。 eval_strategy="epoch" 表示每个 epoch 结束后进行一次评估
    logging_dir="./logs",
    logging_steps=100,
    report_to=["tensorboard"],
)

# 7. 初始化Trainer
trainer = Trainer(
    model=model,                                    # 要训练的模型
    args=training_args,                            # 训练参数配置
    train_dataset=tokenized_datasets["train"],     # 训练数据集
    eval_dataset=tokenized_datasets["validation"], # 验证数据集
    data_collator=data_collator,                  # 数据整理器（处理批次数据）
    compute_metrics=compute_metrics,               # 评估函数
)

# 8. 训练模型
print("\n开始训练模型...")
trainer.train()


# 1.compute_metrics的自信逻辑是每一轮训练结束之后都取出datasets内的validation数据库传入到compute_metrics方法内进行数据集验证
# 2.Trainer内传递的eval_dataset对象就是训练数据集的tokenized_datasets["validation"]，他的数据结构一定是与训练时所用的结构一致
# 3.compute_metrics比较重要 我在方法内部加了注释


##### training log example #####
# {'loss': 0.514, 'grad_norm': 8.539349555969238, 'learning_rate': 2.824981844589688e-05, 'epoch': 1.31}
# {'loss': 0.4553, 'grad_norm': 24.170791625976562, 'learning_rate': 2.4618736383442268e-05, 'epoch': 1.53}
# {'loss': 0.4115, 'grad_norm': 57.1841926574707, 'learning_rate': 2.0987654320987655e-05, 'epoch': 1.74}
# {'loss': 0.4301, 'grad_norm': 12.096426963806152, 'learning_rate': 1.7356572258533045e-05, 'epoch': 1.96}
# {'eval_loss': 0.4542405307292938, 'eval_accuracy': 0.8406862745098039, 'eval_f1': 0.889267461669506, 'eval_runtime': 8.4326, 'eval_samples_per_second': 48.384, 'eval_steps_per_second': 6.048, 'epoch': 2.0}
# {'loss': 0.3063, 'grad_norm': 43.397682189941406, 'learning_rate': 1.3725490196078432e-05, 'epoch': 2.18}
# {'loss': 0.3288, 'grad_norm': 1.4191267490386963, 'learning_rate': 1.009440813362382e-05, 'epoch': 2.4}
# {'loss': 0.3381, 'grad_norm': 0.46658429503440857, 'learning_rate': 6.463326071169208e-06, 'epoch': 2.61}
# {'loss': 0.28, 'grad_norm': 1.2106271982192993, 'learning_rate': 2.832244008714597e-06, 'epoch': 2.83}
# {'eval_loss': 0.5922794938087463, 'eval_accuracy': 0.8382352941176471, 'eval_f1': 0.8877551020408163, 'eval_runtime': 5.187, 'eval_samples_per_second': 78.658, 'eval_steps_per_second': 9.832, 'epoch': 3.0}
# {'train_runtime': 437.8544, 'train_samples_per_second': 25.132, 'train_steps_per_second': 3.145, 'train_loss': 0.4564213610768924, 'epoch': 3.0}