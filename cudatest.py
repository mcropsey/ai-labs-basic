from transformers import pipeline
import torch

# Select GPU if available
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline("sentiment-analysis", device=device)

result = classifier("WSL with GPU should be blazing fast!")[0]
print(f"Label: {result['label']}, with score: {result['score']:.2f}")
