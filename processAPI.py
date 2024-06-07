from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import json
import processSanskrit
from flask_cors import cross_origin  # Add this import


app = Flask(__name__)
CORS(app)
app.debug = True


@app.route('/process', methods=['POST'])
@cross_origin()
def process_text():
    text = request.data.decode('utf-8')
    processed_text = processSanskrit.process(text)  # Print the output to the console
    print
    return jsonify(processed_text) # Convert the response to a string

@app.route('/dict_entry', methods=['POST'])
@cross_origin()
def get_dict_entry():
    data = request.get_json()
    word = data.get('word')
    if word is not None:
        entry = processSanskrit.get_voc_entry([word])
        return jsonify(entry)
    else:
        return jsonify({'error': 'Missing word'}), 400

@app.route('/transliterate', methods=['POST'])
@cross_origin()
def transliterate_text():
    data = request.get_json()
    text = data.get('text')
    transliteration_scheme = data.get('transliteration_scheme')
    if text is not None and transliteration_scheme is not None:
        processed_text = processSanskrit.transliterateAnything(text, transliteration_scheme)
        return jsonify(processed_text)
    else:
        return jsonify({'error': 'Missing text or transliteration_scheme'}), 400



if __name__ == '__main__':
    app.run(port=5000)


