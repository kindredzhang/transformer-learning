import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def demonstrate_transformer_concepts():
    # 1. 基本设置
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    # 2. 准备示例文本
    sequence1 = "I've been waiting for a HuggingFace course my whole life."
    sequence2 = "I hate this so much!"
    
    # 3. 单个序列处理示例
    print("\n=== 单个序列处理 ===")
    # 手动分步处理（教学目的）
    tokens = tokenizer.tokenize(sequence1)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([ids])
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {ids}")
    print(f"Input tensor shape: {input_tensor.shape}")

    # 使用tokenizer直接处理（实际应用推荐）
    inputs = tokenizer(sequence1, return_tensors="pt")
    print(f"Tokenizer direct output: {inputs}")

    # 4. 批处理示例
    print("\n=== 批处理示例 ===")
    # 4.1 不使用padding的批处理（当句子长度不同时会失败）
    try:
        sequences = [sequence1, sequence2]
        # 这里可能会失败，因为序列长度不同
        tokens_batch = [tokenizer.tokenize(seq) for seq in sequences]
        ids_batch = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_batch]
        print(f"Batch without padding: {ids_batch}")
    except Exception as e:
        print(f"批处理失败（预期的）: {e}")

    # 4.2 使用padding的批处理（推荐方式）
    # padding=True: 将所有序列填充到批次中最长序列的长度 
    # truncation=True: 如果序列超过模型最大长度则截断
    # return_attention_mask=True: 返回attention mask
    batch_inputs = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    print("\n批处理结果:")
    print(f"Input IDs shape: {batch_inputs['input_ids'].shape}")
    print(f"Attention mask shape: {batch_inputs['attention_mask'].shape}")
    print(f"Input IDs:\n{batch_inputs['input_ids']}")
    print(f"Attention mask:\n{batch_inputs['attention_mask']}")

    # 5. 模型推理
    print("\n=== 模型推理 ===")
    # 5.1 单个序列
    with torch.no_grad():
        single_output = model(**tokenizer(sequence1, return_tensors="pt"))
        print(f"Single sequence logits: {single_output.logits}")

    # 5.2 批处理推理
    with torch.no_grad():
        batch_output = model(**batch_inputs)
        print(f"Batch logits:\n{batch_output.logits}")

    # 6. 处理长序列示例
    print("\n=== 长序列处理 ===")
    long_sequence = sequence1 * 10  # 重复10次制造长序列
    # max_length参数控制截断
    long_inputs = tokenizer(
        long_sequence,
        padding=True,
        truncation=True,
        max_length=512,  # 大多数transformer模型的限制是512或1024
        return_tensors="pt"
    )
    print(f"Original sequence length: {len(long_sequence)}")
    print(f"Truncated input IDs length: {long_inputs['input_ids'].shape[1]}")

if __name__ == "__main__":
    demonstrate_transformer_concepts()
