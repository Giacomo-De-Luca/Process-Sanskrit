import os
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import re

MODEL_NAME = "chronbmm/sanskrit5-multitask"
MODEL_DIR = "models/chronbmm_sanskrit5-multitask"
MAX_LENGTH = 512

# Global variables to store loaded model and tokenizer
_model = None
_tokenizer = None
_device = None

def download_model():
    """
    Download the model and tokenizer into the 'models' directory if not already present.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    # If there's nothing saved yet, download from Hugging Face
    if not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
        model_local = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer_local = AutoTokenizer.from_pretrained(MODEL_NAME)
        model_local.save_pretrained(MODEL_DIR)
        tokenizer_local.save_pretrained(MODEL_DIR)
        print(f"Model downloaded and saved to: {MODEL_DIR}")

def load_model():
    """
    Load the T5 model and tokenizer from the 'models' directory.
    """
    global _model, _tokenizer, _device
    
    # Only load if not already loaded
    if _model is None or _tokenizer is None:
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
            
        print(f"Loading model to {_device}...")
        _model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model.to(_device)
        print("Model loaded!")
        
    return _model, _tokenizer, _device


import re


tags_to_remove = {"cp", "snf", "SNNe", "PNF", "SNM", "U", "SGNe", "SBM", "SNF", "ḷ"} # Set for efficient lookup


def process_result(text):
    # Replace underscores with spaces
    text = text.replace("_", " ")
    # Replace non-alphanumeric and non-apostrophe characters outside the BMP with spaces
    # Replace only the invalid combining diacritics within the BMP with spaces
    #text = re.sub(r"[^\w\s'‘’´`]", ' ', text)

    
    ##questi non so se tenerli dentro
    #text = text.replace("  ", " ").strip()

    
    words = text.split() # Split into words based on whitespace
    # Filter out the specific tag words
    filtered_words = [word for word in words if word not in tags_to_remove]
    # Join the remaining words back together
    text = " ".join(filtered_words)

    return text


def run_inference(sentences, mode="segmentation", batch_size=20):
    """
    Given a list of Sanskrit sentences and a mode:
      - 'segmentation'
      - 'segmentation-morphosyntax'
      - 'lemma'
      - 'lemma-morphosyntax'
    Return the list of processed outputs from the model.
    
    Parameters:
    - sentences: List of Sanskrit text to process
    - mode: Processing mode
    - batch_size: Number of sentences to process at once (higher = more efficient)
    """
    prefix_map = {
        "segmentation": "S ",
        "segmentation-morphosyntax": "SM ",
        "lemma": "L ",
        "lemma-morphosyntax": "LM ",
        "segmentation-lemma-morphosyntax": "SLM "
    }

    model, tokenizer, device = load_model()
    prefix = prefix_map.get(mode, "S ")  # Default to lemma if invalid mode
    
    # Process in batches for efficiency
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        input_texts = [f"{prefix}{text}" for text in batch]

        inputs = tokenizer(
            input_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=MAX_LENGTH)
        
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    if mode == "segmentation":
        # Process the results to remove unwanted tags
        processed_results = [process_result(text) for text in results]        
        return processed_results

    return results
