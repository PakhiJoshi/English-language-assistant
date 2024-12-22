from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from googletrans import Translator
import language_tool_python
import os

app = Flask(__name__)

# Load Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()
grammar_tool = language_tool_python.LanguageTool('en-US')

# Routes

@app.route('/')
def home():
    return render_template('index.html')

# 1. Summarization
@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]['summary_text']})

# 2. Translation
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "es")  # Default to Spanish
    if not text:
        return jsonify({"error": "Text is required"}), 400
    translated = translator.translate(text, dest=target_lang)
    return jsonify({"translated_text": translated.text})

# 3. Grammar Checking
@app.route('/check_grammar', methods=['POST'])
def check_grammar():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    matches = grammar_tool.check(text)
    corrections = [{"message": match.message, "offset": match.offset, "length": match.errorLength} for match in matches]
    return jsonify({"corrections": corrections})

if __name__ == "__main__":
    # Make sure the app runs on Windows
    app.run(debug=True, host="127.0.0.1", port=5000)
