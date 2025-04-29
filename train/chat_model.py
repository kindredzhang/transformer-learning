from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import platform

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatModel:
    def __init__(self, model_path="./finetuned_bart"):
        """
        Initialize the chat model
        Args:
            model_path: Path to the fine-tuned model
        """
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            # 检测设备类型并配置
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Metal Performance Shaders) for M2 chip")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")
            
            # 将模型移动到相应设备
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
            # 设置模型为评估模式
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def generate_response(self, prompt, max_length=128, temperature=0.7):
        """
        Generate a response
        Args:
            prompt: User input question
            max_length: Maximum length of generated response
            temperature: Controls randomness (higher = more random)
        Returns:
            str: Generated response
        """
        try:
            # Add prefix
            input_text = f"dialogue: {prompt}"
            
            # Encode input
            inputs = self.tokenizer(
                input_text,
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            # 使用 torch.no_grad() 来减少内存使用
            with torch.no_grad():
                # Generate response with BART-specific parameters
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

# Usage example
if __name__ == "__main__":
    try:
        # 打印系统信息
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
        
        # Create model instance
        chat_model = ChatModel()
        
        # Test some dialogues
        test_prompts = [
            "Hello, who are you?",
            "What can you do?",
            "Tell me a joke",
            "What is artificial intelligence?",
            "How to learn programming?",
            "Recommend some good books"
        ]
        
        print("Starting dialogue model test:")
        print("-" * 50)
        
        for prompt in test_prompts:
            print(f"Question: {prompt}")
            response = chat_model.generate_response(prompt)
            print(f"Answer: {response}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}") 