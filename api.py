from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Global variables for models
models = {}
tokenizers = {}

def get_model(lang):
    global models, tokenizers
    if lang not in models:
        print(f"Loading model for lang: {lang}...")
        model_name = "yuchuantian/AIGC_detector_env3" if lang == "en" else "yuchuantian/AIGC_detector_zhv3"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval() # Set to evaluation mode
            tokenizers[lang] = tokenizer
            models[lang] = model
            print(f"Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None, None
            
    return tokenizers[lang], models[lang]

@app.route('/api/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400
        
    text = data.get('text', '')
    lang = data.get('lang', 'en') # default to english
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    tokenizer, model = get_model(lang)
    if not tokenizer or not model:
        return jsonify({'error': 'Failed to load model for the specified language'}), 500
    
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # According to common AIGC models from this paper, class 1 is AI, class 0 is Human.
            # We will verify this, but typically:
            human_prob = probs[0][0].item()
            ai_prob = probs[0][1].item()
            
            # Return result
            result = {
                'human_probability': human_prob,
                'ai_probability': ai_prob,
                'is_ai': ai_prob > 0.5,
                'lang': lang
            }
            return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    # Pre-load english model on startup
    get_model('en')
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
