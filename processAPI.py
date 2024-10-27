from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import processSanskrit
from flask_cors import cross_origin 
import logging
from datetime import datetime
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DATABASE_API_KEY")


app = Flask(__name__)
CORS(app)
app.debug = True

logging.basicConfig(filename='error.log', level=logging.ERROR)

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

## get a JSON array with two keys:  'text' and 'transliteration_scheme'

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

client = OpenAI(
api_key=api_key)
responses = {}

def API_call (context, content):    
    response = client.chat.completions.create(
        model="gpt-4-O",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": f"{content}"}
        ],
    temperature=0.7,
    max_tokens=2000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    responses = response.choices[0].message.content
    print(responses)
    return responses
    
context = """
You are a medieval Sanskrit expert tasked to translate the following texts from Sanskrit to English. Try to be as literal as possible. If technical terms are encountered, such as the names of Mudra or Pranayama techniques, leave the names untranslated. In case of more common technical terms such as 'Prakṛti', it is also acceptable to translate them, leaving the original term between parentheses. 

Provide a translation of the text, and return it in a phrase-by-phrase JSON format. Avoid splitting sentences in the middle. Ensure that each phrase is a complete, grammatically correct unit, and not just a part of a sentence. In case there are '|' or '||' characters in the text, they should be treated as line breaks, as they delimit the end of a verse or a paragraph.

The JSON should be an array of objects, wrapped inside an object with the 'translation' property, containing the translation. Inside the array, each phrase should have its own object, having as properties 'Sanskrit', with the original text for the phrase, and 'English', with the English translated text for the phrase.

Here's an example of the expected format:
{
    "translation": [
        {
            "Sanskrit": "सर्वे भवन्तु सुखिनः।",
            "English": "May all be happy."
        },
        {
            "Sanskrit": "सर्वे सन्तु निरामयाः।",
            "English": "May all be free from illness."
        }
    ]
}"""

def fix_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # If there's an error, add a closing quote and try again
        fixed_json_string = json_string + '"'
        return json.loads(fixed_json_string)

@app.route('/translate', methods=['POST'])
@cross_origin()
def translate_text():
    content = request.data.decode('utf-8')
    responses = API_call(context, content) 

    # Create a unique filename using the current time
    filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.json')

    # Write the JSON object to a file
    with open(filename, 'w') as f:
        json.dump(responses, f)

    return Response(responses, mimetype='application/json')


@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    app.logger.error(f"An error occurred: {str(e)}")
    
    # Return a string of the exception to the client
    return str(e), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=False)


