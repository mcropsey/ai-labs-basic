from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I hate my job")
print(result)
