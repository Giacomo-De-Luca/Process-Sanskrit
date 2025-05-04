import os
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import re


# --- Configuration ---
MODEL_NAME = "chronbmm/sanskrit5-multitask" # Just the HF Hub name
MAX_LENGTH = 512
# --- End Configuration ---

# --- Global variables ---
_model = None
_tokenizer = None
_device = None
# --- End Global variables ---

def load_model():
    """
    Loads the T5 model and tokenizer using the Hugging Face cache.
    Downloads the model automatically on the first run if not cached.
    """
    global _model, _tokenizer, _device

    if _model is None or _tokenizer is None:
        # Determine device
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")

        print(f"Loading model '{MODEL_NAME}'...")
        print(f" (Using Hugging Face cache: ~/.cache/huggingface/hub or HF_HOME)") # Inform user

        try:
            # Let transformers handle download/cache check automatically
            # Pass cache_dir=... here only if you *really* need a non-default location
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            _model.to(_device)
            print(f"Model loaded successfully to {_device}!")

        except OSError as e:
            print(f"\n!!! Error loading model '{MODEL_NAME}' from Hugging Face Hub: {e}", file=sys.stderr)
            print("!!! Please check your internet connection and the model name.", file=sys.stderr)
            print("!!! You might need to log in using `huggingface-cli login` if it's a private model.", file=sys.stderr)
            # Decide how to handle failure: raise error, return None, etc.
            raise  # Re-raise the error to stop execution if model is critical
        except Exception as e:
             print(f"\n!!! An unexpected error occurred loading the model: {e}", file=sys.stderr)
             raise


    return _model, _tokenizer, _device


### Set of tags to remove from the output
tags_to_remove = {"cp", "snf", "SNNe", "PNF", "SNM", "U", "SGNe", "SBM", "SNF", "ḷ", "ḷ"} # Set for efficient lookup


def process_result(text):
    # Replace underscores with spaces
    text = text.replace("_", " ")
    # Replace non-alphanumeric and non-apostrophe characters outside the BMP with spaces
    # Replace only the invalid combining diacritics within the BMP with spaces
    #text = re.sub(r"[^\w\s'‘’´`]", ' ', text)
    
    ##questi non so se tenerli dentro
    #text = text.replace("  ", " ").strip()
    #     
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
