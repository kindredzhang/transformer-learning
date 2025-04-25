from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# 方法1：使用预训练的翻译模型
def test_pretrained_translator():
    print("\n=== 使用预训练翻译模型 ===\n")
    
    # 加载预训练的英文到中文翻译模型
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    translator = pipeline("translation", model=model_name)
    
    # 测试数据
    test_sentences = [
        "Hello, my name is John.",
        "I want to learn Chinese.",
        "This is a beautiful day.",
        "How much does this cost?",
        "See you tomorrow!"
    ]
    
    for text in test_sentences:
        result = translator(text, max_length=50)
        print(f"英文: {text}")
        print(f"中文翻译: {result[0]['translation_text']}")
        print("-" * 50)

# 方法2：使用我们微调的模型
def test_finetuned_model():
    print("\n=== 使用我们微调的模型 ===\n")
    
    # 加载微调后的模型
    model_path = "./finetuned_bart"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # 测试数据
    test_sentences = [
        "Translate this to Chinese: 'Hello, my name is John.'",
        "Translate this to Chinese: 'I want to learn Chinese.'",
        "Translate this to Chinese: 'This is a beautiful day.'"
    ]
    
    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输入: {text}")
        print(f"输出: {translation}")
        print("-" * 50)

# 运行测试
if __name__ == "__main__":
    try:
        test_pretrained_translator()
    except Exception as e:
        print(f"预训练模型测试失败: {e}")
    
    try:
        test_finetuned_model()
    except Exception as e:
        print(f"微调模型测试失败: {e}")
