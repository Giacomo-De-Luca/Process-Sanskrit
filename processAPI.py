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

if __name__ == '__main__':
    app.run(port=5000)


