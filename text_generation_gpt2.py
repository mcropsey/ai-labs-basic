from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model="gpt2", device=device)

prompt = "In the future, artificial intelligence will"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]["generated_text"])
