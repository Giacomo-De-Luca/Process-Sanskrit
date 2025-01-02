
from utils.loadResources import mwdictionaryKeys
from utils.lexicalResources import SANSKRIT_PREFIXES, SANDHI_VARIATIONS, VOWEL_SANDHI_INITIALS
from functions.rootAnyWord import root_any_word
from typing import Union, List

def try_match_with_prefixes(word, debug=False):
    """
    Attempts to match a word by checking for prefixes ONLY at the start.
    Does not recursively look for prefixes in the remainder.
    
    Args:
        word: The word to analyze in SLP1 notation
        debug: Whether to print debug information
    
    Returns:
        tuple: (matched_word, end_letter) or None if no match found
    """
    if debug:
        print(f"\nTrying to match word with prefixes: {word}")
    
    # First try the whole word as-is
    if word in mwdictionaryKeys:
        if debug:
            print(f"Found direct match in dictionary: {word}")
        return (word, word[-1])
    
    # Look for prefixes only at the start
    for prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
        if word.startswith(prefix):
            remainder = word[len(prefix):]
            if debug:
                print(f"Found prefix {prefix}, trying remainder: {remainder}")
            
            # For the remainder, just try a direct dictionary match
            if remainder in mwdictionaryKeys:
                if debug:
                    print(f"Found remainder in dictionary: {remainder}")
                return (word[:len(prefix) + len(remainder)], word[len(prefix) + len(remainder) - 1])    
    
    if debug:
        print("No match found after prefix analysis")
    return None

def dict_word_iterative(word, debug=False):
    """
    Dictionary lookup that integrates prefix matching with sandhi variations.
    For each possible word length, tries:
    1. Direct dictionary match
    2. Prefix matches
    3. Sandhi variations (with prefix checks on each variation)
    """
    temp_word = word
    best_match = None
    best_length = 0
    
    if debug:
        print(f"Attempting to match word: {word}")
    
    # First try root_any_word on complete word
    root_result = root_any_word(temp_word)
    if root_result:
        if debug:
            print(f"Found inflected form: {temp_word}")
        return (temp_word, temp_word[-1])
    
    while temp_word and len(temp_word) > 1:
        # Try direct dictionary match
        if temp_word in mwdictionaryKeys:
            if len(temp_word) > best_length:
                if debug:
                    print(f"Found dictionary match: {temp_word}")
                best_match = temp_word
                best_length = len(temp_word)
        
        # Try prefix matches on current word
        prefix_match = try_match_with_prefixes(temp_word, debug)
        if prefix_match and len(prefix_match[0]) > best_length:
            if debug:
                print(f"Found prefix match: {prefix_match[0]}")
            best_match = prefix_match[0]
            best_length = len(prefix_match[0])
        
        # Try sandhi variations and check prefixes on each variant
        last_char = temp_word[-1]
        if last_char in SANDHI_VARIATIONS:
            for variant in SANDHI_VARIATIONS[last_char]:
                test_word = temp_word[:-1] + variant
                
                # Try direct match of sandhi variant
                if test_word in mwdictionaryKeys:
                    if len(test_word) > best_length:
                        if debug:
                            print(f"Found match with sandhi variation: {test_word}")
                        best_match = test_word
                        best_length = len(test_word)
                
                # Try prefix match on sandhi variant
                prefix_match = try_match_with_prefixes(test_word, debug)
                if prefix_match and len(prefix_match[0]) > best_length:
                    if debug:
                        print(f"Found prefix match on sandhi variant: {prefix_match[0]}")
                    best_match = prefix_match[0]
                    best_length = len(prefix_match[0])
        
        # If we found a match of full length, stop here
        if best_match and len(best_match) == len(temp_word):
            break
            
        temp_word = temp_word[:-1]
    
    if best_match:
        return (best_match, word[len(best_match)-1])
    
    if debug:
        print("No match found")
    return None



def root_compounds(word, debug=False, inflection=True, ):
    """
    Analyzes a long Sanskrit compound with improved sandhi handling between segments.
    """
    if debug:
        print("\nStarting analysis of:", word)
        print("Length:", len(word))

        # Handle initial apostrophe (avagraha)
    if word.startswith("'"):
        word = 'a' + word[1:]
        
    roots = []
    current_pos = 0
    
    while current_pos < len(word):
        remaining = word[current_pos:]
        if debug:
            print(f"\nAnalyzing segment starting at position {current_pos}: {remaining}")
            
        # Try the base case first
        best_match = None
        best_length = 0
        first_match = dict_word_iterative(remaining)

        
        if first_match:
            best_match = first_match
            best_length = len(first_match[0])
            if debug:
                print(f"Found base match: {first_match[0]} (ends with {first_match[1]})")
        
        # If the previous match ended in a vowel that can cause sandhi,
        # try analyzing the remaining text with added initial vowels
        if current_pos > 0:
            prev_end = word[current_pos - 1]
            if prev_end in VOWEL_SANDHI_INITIALS:
                if debug:
                    print(f"Trying sandhi variations for previous ending {prev_end}")
                
                for initial_vowel in VOWEL_SANDHI_INITIALS[prev_end]:
                    test_word = initial_vowel + remaining
                    if debug:
                        print(f"Trying with added {initial_vowel}: {test_word}")

                    # First try root_any_word with the sandhi-modified version
                    root_result = root_any_word(test_word)
                    if root_result:
                        if debug:
                            print(f"Found inflected form with sandhi: {test_word}")
                        test_match = (test_word, test_word[-1])
                        if len(test_word) > best_length:
                            best_match = test_match
                            best_length = len(test_word)
                            continue
                    
                    # If no inflected form found, try dictionary match
                    test_match = dict_word_iterative(test_word)
                    if test_match and len(test_match[0]) > best_length:
                        best_match = test_match
                        best_length = len(test_match[0])
                        if debug:
                            print(f"Found better match with sandhi: {test_match[0]}")


            if remaining.startswith('ch'):
                test_word = 'ś' + remaining[1:]
                if debug:
                    print(f"Trying with S instead of ch: {test_word}")
                test_match = dict_word_iterative(test_word)
                if test_match and len(test_match[0]) > best_length:
                    best_match = test_match
                    best_length = len(test_match[0])
                    if debug:
                        print(f"Found better match with S: {test_match[0]}")
        
        if not best_match:
            if debug:
                print("No match found, moving forward one character")
            current_pos += 1
            continue
            
        matched_word, end_letter = best_match
        
        if inflection==True:
            # Process the matched word
            root_entry = root_any_word(matched_word)
            if root_entry:
                if isinstance(root_entry, list):
                    roots.extend(root_entry)
                else:
                    roots.append(root_entry)
            else:
                roots.append(matched_word)
        else:
            roots.append(matched_word)
        
        # Adjust position based on whether we used an added vowel
        vowel_adjustment = 1 if (current_pos > 0 and 
                                word[current_pos - 1] in VOWEL_SANDHI_INITIALS and 
                                matched_word[0] in VOWEL_SANDHI_INITIALS[word[current_pos - 1]]) else 0
        current_pos += max(len(matched_word) - vowel_adjustment, 1)

        if debug:
            print(f"Advanced position by {len(matched_word) - vowel_adjustment}")
    
    return roots

def process_root_result(root_result: Union[List, str]) -> str:
    """
    Process a single element from root_compounds output into a simple string.
    
    Parameters:
    - root_result: Either a string or a list containing morphological analysis
    
    Returns:
    - str: The base form of the word
    """
    if isinstance(root_result, str):
        return root_result
    # If it's a list, the last element is usually the base form
    # Lists from root_compounds typically end with the base form
    return root_result[0]
