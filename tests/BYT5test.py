import os
import json
import re
import logging
from typing import List, Any

## module to test the BYT5 model on entire books. 
## this module can't actually word unless there is a JSON version of the GRETIL library locally. 
## the script to convert the Gretil library (or in general the Sanskrit text libraries) to JSON will be released later
## as the current version is working, - but still needs major works to handle the lack of standardisation of the TEI encoding. 

# Import functionality from model_inference.py
from model_inference import download_model, load_model, run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_specific_file(corpus_folder: str, file_name: str) -> List[str]:
    """
    Extract all Sanskrit text from a specific GRETIL JSON file.
    
    Args:
        corpus_folder: Path to folder containing GRETIL JSON files
        file_name: Name of the specific file to process
        
    Returns:
        List[str]: List of extracted text strings
    """
    file_path = os.path.join(corpus_folder, file_name)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract all text using the recursive function
        text_list = extract_all_text(data)
        
        # Clean the extracted text
        clean_texts = []
        for text in text_list:
            # Clean the line of any metadata or verse numbers
            cleaned_text = re.sub(r'\|\|\s*[A-Z_0-9.]+\s*\|\|', '', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            if cleaned_text:  # Only add non-empty strings
                clean_texts.append(cleaned_text)
                
        return clean_texts
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []

def extract_all_text(node: Any, text_list: List[str] = None) -> List[str]:
    """
    Recursively extract all text from JSON structure.
    This is adapted from the CompoundExtractor class.
    
    Args:
        node: Current JSON node
        text_list: List to accumulate text
        
    Returns:
        List[str]: List of extracted text strings
    """
    if text_list is None:
        text_list = []
        
    if isinstance(node, dict):
        if "text" in node and isinstance(node["text"], str):
            text_list.append(node["text"])
        
        for value in node.values():
            if isinstance(value, (dict, list)):
                extract_all_text(value, text_list)
                
    elif isinstance(node, list):
        for item in node:
            extract_all_text(item, text_list)
            
    return text_list

def process_yogasutra(corpus_folder: str, max_samples: int = 50):
    """
    Extract text from the Yoga Sutra file and run the segmentation model on it.
    
    Args:
        corpus_folder: Path to the folder containing the GRETIL JSON files
        max_samples: Maximum number of text samples to process
    
    Returns:
        tuple: (original_texts, segmented_texts) - Lists of original and processed texts
    """
    file_name = "sa_patañjali-yogasūtra-with-bhāṣya.json"
    
    # Extract text from the specified file
    logger.info(f"Extracting text from {file_name}...")
    texts = extract_text_from_specific_file(corpus_folder, file_name)
    
    if not texts:
        logger.error("No texts were extracted from the file.")
        return [], []
        
    logger.info(f"Extracted {len(texts)} text segments.")
    
    # Limit the number of samples if needed
    if max_samples and max_samples < len(texts):
        logger.info(f"Limiting to {max_samples} samples for processing.")
        texts = texts[:max_samples]
    
    # Ensure the model is downloaded and loaded
    download_model()
    
    # Run the model in segmentation mode
    logger.info("Running segmentation model on extracted texts...")
    segmented_texts = run_inference(texts, mode="segmentation")
    
    return texts, segmented_texts

if __name__ == "__main__":
    # Set the corpus folder path - update this to your actual path
    corpus_folder = "/Users/jack/Documents/GitHub/SanskritTextTest"  
    
    # Process the Yoga Sutra text
    original_texts, segmented_texts = process_yogasutra(corpus_folder, max_samples=20)
    
    # Display the results
    logger.info(f"Processed {len(segmented_texts)} text segments.")
    
    print("\nSample Original and Segmented Texts:")
    for i, (original, segmented) in enumerate(zip(original_texts, segmented_texts)):
        if i >= 5:  # Only show first 5 samples
            break
        print(f"\nSample #{i+1}:")
        print(f"Original: {original[:100]}{'...' if len(original) > 100 else ''}")
        print(f"Segmented: {segmented[:100]}{'...' if len(segmented) > 100 else ''}")
    
    # Optional: Save results to file
    output_file = "yogasutra_segmentation_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (original, segmented) in enumerate(zip(original_texts, segmented_texts)):
            f.write(f"Sample #{i+1}:\n")
            f.write(f"Original: {original}\n")
            f.write(f"Segmented: {segmented}\n\n")
    
    logger.info(f"Results saved to {output_file}")