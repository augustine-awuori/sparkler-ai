from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load fine-tuned model and tokenizer
try:
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('awuori/sparkler-ai-beta-v1.1')
    model = AutoModelForCausalLM.from_pretrained(
        "awuori/sparkler-ai-beta-v1.1", torch_dtype="auto")
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        if not data or 'text' not in data:
            logger.warning("Missing 'text' in request")
            return jsonify({'error': 'Missing "text" in request'}), 400
        input_text = data['text']
        max_length = min(int(data.get('max_length', 100)), 500)

        logger.info(f"Processing query: {input_text}")

        # Tokenize input
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

        # Decode response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Response generated successfully")
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/')
def home():
    return 'Welcome to Sparkler AI (AKA Grao AI) server!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)