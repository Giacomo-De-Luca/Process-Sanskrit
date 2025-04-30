from process_sanskrit.functions.process import process
from process_sanskrit.functions.model_inference import run_inference
from typing import List, Union, Any



def _format_roots_output(processed_word_results: List[Any]) -> str:
    """Helper function to format the output when mode is 'roots'."""
    formatted_words = [] # Store formatted string for each original word's processing result
    for word_output in processed_word_results: # word_output is the List[Union[str, Tuple]] from extract_roots for a single word
        # print(word_output) # Keep or remove debug print as needed
        current_word_parts = []
        if isinstance(word_output, list):
            for item in word_output: # Iterate through items returned by extract_roots for this word
                if isinstance(item, tuple):
                    # Multiple roots found for a sub-part, format as (a | b)
                    current_word_parts.append(f"({' | '.join(item)})")
                else:
                    # Single root found for a sub-part
                    current_word_parts.append(str(item)) # Ensure string
        elif isinstance(word_output, tuple): # Fallback if process returns tuple directly
            current_word_parts.append(f"({' | '.join(word_output)})")
        else: # Fallback if process returns string directly
            current_word_parts.append(str(word_output))

        # Join parts for the current word.
        formatted_words.append(" ".join(current_word_parts))

    # Join all the formatted word strings with spaces
    return " ".join(formatted_words)



def processBYT5(text: Union[str, List[str]], mode="detailed", *dict_names) -> Union[List[Any], Any]:
    """
    Process Sanskrit text using BYT5 model for segmentation and then analyze each word.
    
    Args:
        text: Input text (single string) or list of texts to process
        mode: Processing mode - "none" (default), "roots", or "parts"
        *dict_names: Dictionary names to look up words in
    
    Returns:
        For single string input:
            - If mode="roots": A string of joined root words, with multiple possibilities 
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
            word_result = process(word, mode=mode, *dict_names)
            for root_result in word_result:
                processed_results.append(root_result)
        
        # Join results if mode="roots" was specified
        if mode == "roots":
                return _format_roots_output(processed_results)
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
                word_result = process(word, mode=mode, *dict_names)
                processed_segment.append(word_result)
            
            # Join results if mode="roots" was specified
            if mode == "roots":
                segment_result = _format_roots_output(processed_segment) # Use helper function
                processed_segments.append(segment_result)
            else:
                # If not roots mode, append the list of word results for the segment
                processed_segments.append(processed_segment)
            
        return processed_segments
    
    # Handle invalid input type
    else:
        raise TypeError("Input must be a string or a list of strings")