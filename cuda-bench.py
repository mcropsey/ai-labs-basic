import time
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", device=device)

start = time.time()
for _ in range(5):
    result = classifier("This should be running on GPU!")[0]
    print(f"{result['label']} ({result['score']:.2f})")
end = time.time()

print(f"Total time: {end - start:.2f} seconds")
