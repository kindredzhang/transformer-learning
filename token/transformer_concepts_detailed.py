import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TransformerConceptsDemo:
    def __init__(self):
        """初始化模型和tokenizer"""
        self.checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
        
        # 示例文本
        self.sequence1 = "I've been waiting for a HuggingFace course my whole life."
        self.sequence2 = "I hate this so much!"

    def demonstrate_basic_tokenization(self):
        """演示基本的tokenization过程"""
        print("\n=== 基本tokenization演示 ===")
        # 手动分步处理
        tokens = self.tokenizer.tokenize(self.sequence1)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([ids])
        
        print(f"原始文本: {self.sequence1}")
        print(f"分词结果: {tokens}")
        print(f"Token IDs: {ids}")
        print(f"转换为tensor后的形状: {input_tensor.shape}")

    def demonstrate_batch_processing(self):
        """
        演示批处理和padding的概念
        
        Q1: 为什么多段语句传入模型时必须统一长度？
        A1: 深度学习模型在批处理时需要统一的张量形状才能进行并行计算。
            这是因为神经网络的矩阵运算要求输入维度一致。
        
        Q2: 为什么统一长度后还需要truncation=True？
        A2: - padding是把短序列补齐到最长序列的长度
            - truncation是处理超出模型最大限制的序列
            这是两个不同的问题的解决方案
        """
        print("\n=== 批处理演示 ===")
        sequences = [self.sequence1, self.sequence2]
        
        # 1. 不使用padding的情况（会失败）
        try:
            tokens_batch = [self.tokenizer.tokenize(seq) for seq in sequences]
            print("不使用padding的分词结果:")
            print(f"第一句tokens长度: {len(tokens_batch[0])}")
            print(f"第二句tokens长度: {len(tokens_batch[1])}")
        except Exception as e:
            print(f"不使用padding的批处理失败: {e}")

        # 2. 使用padding的正确方式
        batch_inputs = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        print("\n使用padding的批处理结果:")
        print(f"Input IDs形状: {batch_inputs['input_ids'].shape}")
        print(f"Attention mask形状: {batch_inputs['attention_mask'].shape}")

    def demonstrate_attention_mask(self):
        """
        演示attention mask的作用
        
        Q3: attention_mask的作用是什么？
        A3: attention_mask用于标识哪些token是真实的，哪些是padding的：
            - 1表示真实token
            - 0表示padding token
            这样模型在计算attention时会忽略padding token的影响
        """
        print("\n=== Attention Mask演示 ===")
        sequences = [self.sequence1, "Hi"]  # 故意使用长度差异很大的两句话
        
        outputs = self.tokenizer(
            sequences,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        print("Token IDs:")
        print(outputs['input_ids'])
        print("\nAttention Mask:")
        print(outputs['attention_mask'])
        print("\n1表示真实token，0表示padding token")

    def demonstrate_model_inference(self):
        """
        演示模型推理过程
        
        Q4: model(**batch_inputs)是什么意思？
        A4: **是Python的字典解包操作符，相当于：
            model(
                input_ids=batch_inputs['input_ids'],
                attention_mask=batch_inputs['attention_mask']
            )
        """
        print("\n=== 模型推理演示 ===")
        # 单个序列推理
        with torch.no_grad():
            inputs = self.tokenizer(self.sequence1, return_tensors="pt")
            outputs = self.model(**inputs)
            print(f"单个序列的logits: {outputs.logits}")

    def demonstrate_long_sequence(self):
        """
        演示长序列处理
        
        Q5: max_length=512的作用是什么？
        A5: - 指定序列的最大长度限制
            - 512是很多Transformer模型的默认限制
            - 这个限制来自于：
              * 自注意力机制的计算复杂度O(n²)
              * 位置编码的设计限制
              * 训练时的内存限制
        """
        print("\n=== 长序列处理演示 ===")
        # 创建一个超长序列
        long_sequence = self.sequence1 * 10
        
        # 处理长序列
        long_inputs = self.tokenizer(
            long_sequence,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        print(f"原始序列长度: {len(long_sequence)}")
        print(f"截断后的token长度: {long_inputs['input_ids'].shape[1]}")
        print("注：超过512的部分会被截断")

def main():
    demo = TransformerConceptsDemo()
    demo.demonstrate_basic_tokenization()
    demo.demonstrate_batch_processing()
    demo.demonstrate_attention_mask()
    demo.demonstrate_model_inference()
    demo.demonstrate_long_sequence()

if __name__ == "__main__":
    main()