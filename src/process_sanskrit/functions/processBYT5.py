from processSanskrit import process
from process_sanskrit.functions.model_inference import run_inference
from typing import List, Union, Any

def processBYT5(text: Union[str, List[str]], roots="none", *dict_names) -> Union[List[Any], Any]:
    """
    Process Sanskrit text using BYT5 model for segmentation and then analyze each word.
    
    Args:
        text: Input text (single string) or list of texts to process
        roots: Processing mode - "none" (default), "roots", or "parts"
        *dict_names: Dictionary names to look up words in
    
    Returns:
        For single string input:
            - If roots="roots": A string of joined root words, with multiple possibilities 
              for a word joined in parentheses like "(bhāva | bhā)"
            - Otherwise: A list of processed results for each word
        
        For list input:
            - List of processed results for each text segment
    """
    # Handle single string input
    if isinstance(text, str):
        # Run segmentation on the single string
        segmented_texts = run_inference([text], mode="segmentation", batch_size=1)
        
        if not segmented_texts:
            return []
            
        # Get the word list from the segmented text
        word_list = segmented_texts[0].split()
        
        # Process each word
        processed_results = []
        for word in word_list:
            # Call process() with all arguments
            word_result = process(word, roots=roots, *dict_names)
            processed_results.append(word_result)
        
        # Join results if roots="roots" was specified
        if roots == "roots":
            formatted_results = []
            for result in processed_results:
                # Check if the result is a list with multiple options
                if isinstance(result, list) and len(result) > 1:
                    # Format multiple possibilities with parentheses and pipe separator
                    formatted_result = f"({' | '.join(result)})"
                elif isinstance(result, list) and len(result) == 1:
                    # Extract single item from list
                    formatted_result = result[0]
                else:
                    # Use as is (should be a string)
                    formatted_result = result
                formatted_results.append(formatted_result)
            
            # Join all the formatted results with spaces
            return " ".join(formatted_results)
        
        return processed_results
    
    # Handle list of strings input
    elif isinstance(text, list):
        # Run segmentation on the list of texts
        segmented_texts = run_inference(text, mode="segmentation", batch_size=20)
        
        # Process each segment
        processed_segments = []
        for segment_text in segmented_texts:
            word_list = segment_text.split()
            
            # Process each word in the segment
            processed_segment = []
            for word in word_list:
                word_result = process(word, roots=roots, *dict_names)
                processed_segment.append(word_result)
            
            # Join results if roots="roots" was specified
            if roots == "roots":
                formatted_results = []
                for result in processed_segment:
                    # Check if the result is a list with multiple options
                    if isinstance(result, list) and len(result) > 1:
                        # Format multiple possibilities with parentheses and pipe separator
                        formatted_result = f"({' | '.join(result)})"
                    elif isinstance(result, list) and len(result) == 1:
                        # Extract single item from list
                        formatted_result = result[0]
                    else:
                        # Use as is (should be a string)
                        formatted_result = result
                    formatted_results.append(formatted_result)
                
                # Join all the formatted results with spaces
                processed_segment = " ".join(formatted_results)
                
            processed_segments.append(processed_segment)
            
        return processed_segments
    
    # Handle invalid input type
    else:
        raise TypeError("Input must be a string or a list of strings")