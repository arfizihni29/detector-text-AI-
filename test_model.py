from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "yuchuantian/AIGC_detector_env3"
print(f"Loading {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Model loaded successfully.")
    
    texts = [
        "This is a sample human written text, which is supposed to be recognized as human.",
        "As an AI language model, I am designed to assist you with a variety of tasks."
    ]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
    for i, text in enumerate(texts):
        print(f"Text: {text}")
        print(f"Probs: {probs[i].tolist()}")
        print(f"Predicted class: {torch.argmax(probs[i]).item()}")
        print("-" * 30)
except Exception as e:
    print(f"Error: {e}")
