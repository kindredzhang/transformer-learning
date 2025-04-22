from transformers import pipeline

# print("\n=== 1. 情感分析 (Sentiment Analysis) ===\n")

# 创建情感分析pipeline
# sentiment_classifier = pipeline("sentiment-analysis")

# 单个文本分析
# result = sentiment_classifier("I've been waiting for a HuggingFace course my whole life.")
# print("单个文本分析结果:")
# print(result)

# # 多个文本分析
# results = sentiment_classifier([
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!"
# ])
# print("\n多个文本分析结果:")
# print(results)

print("\n=== 2. 零样本分类 (Zero-shot Classification) ===\n")

# 创建零样本分类pipeline
zero_shot_classifier = pipeline("zero-shot-classification")

# 零样本分类示例
result = zero_shot_classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print("零样本分类结果:")
print(f"文本: {result['sequence']}")
print(f"标签: {result['labels']}")
print(f"分数: {result['scores']}")

# print("\n=== 3. 文本生成 (Text Generation) ===\n")

# 创建文本生成pipeline
# generator = pipeline("text-generation", model="distilgpt2")
# result = generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
# )
# # 文本生成示例
# # result = text_generator("Hugging Face is", max_length=30, num_return_sequences=2)
# print("文本生成结果:")
# for i, generated_text in enumerate(result):
#     print(f"生成文本 {i+1}: {generated_text['generated_text']}")

# print("\n=== 4. fill mask ===\n")

# 创建file mask pipeline
# unmasker = pipeline("fill-mask")
# result = unmasker("This course will teach you all about <mask> models.", top_k=2)
# print("fill mask结果:")
# print(result)


# named entity recognition (NER)
# print("\n=== 5. 命名实体识别 (NER) ===\n")
# ner = pipeline("ner", grouped_entities=True)
# result = ner("My name is Kindred and I work at Moozumni in China Wuhan.")
# print("命名实体识别结果:")
# print(result)


# summarization
# print("\n=== 6. 摘要生成 (summarization) ===\n")
# summarizer = pipeline("summarization")
# result = summarizer(
#     """
#     America has changed dramatically during recent years. Not only has the number of 
#     graduates in traditional engineering disciplines such as mechanical, civil, 
#     electrical, chemical, and aeronautical engineering declined, but in most of 
#     the premier American universities engineering curricula now concentrate on 
#     and encourage largely the study of engineering science. As a result, there 
#     are declining offerings in engineering subjects dealing with infrastructure, 
#     the environment, and related issues, and greater concentration on high 
#     technology subjects, largely supporting increasingly complex scientific 
#     developments. While the latter is important, it should not be at the expense 
#     of more traditional engineering.

#     Rapidly developing economies such as China and India, as well as other 
#     industrial countries in Europe and Asia, continue to encourage and advance 
#     the teaching of engineering. Both China and India, respectively, graduate 
#     six and eight times as many traditional engineers as does the United States. 
#     Other industrial countries at minimum maintain their output, while America 
#     suffers an increasingly serious decline in the number of engineering graduates 
#     and a lack of well-educated engineers.
# """
# )
# print("摘要生成结果:")
# print(result)


# # translation
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# result = translator("Ce cours est produit par Hugging Face.")
# print("翻译结果:")
# print(result)
