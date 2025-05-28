from .sandhiSplitScorer import scorer
# utils/sandhi_splitter.py
from typing import List, Tuple, Dict, Union, Optional
import ast
from dataclasses import dataclass
from sanskrit_parser import Parser

## cache is currently not used and commented out


parser = Parser(output_encoding='iast')

@dataclass
class SplitResult:
    """Class to hold the result of a sandhi split with scoring information"""
    split: List[str]
    score: float
    subscores: dict
    all_splits: Optional[List[Tuple[List[str], float, dict]]] = None

def sandhi_splitter(
    text_to_split: str, 
    cached: bool = False, 
    attempts: int = 10,
    detailed_output: bool = False
) -> List[str]:
    """
    Enhanced sandhi splitter that returns the best split by default.
    
    Parameters:
    - text_to_split (str): The text to split
    - cached (bool): Whether to use caching
    - attempts (int): Number of splitting attempts to try
    - detailed_output (bool): If True, returns tuple (split, score, subscores, all_splits)
    
    Returns:
    - List[str]: The best split by default
    - If detailed_output=True: Tuple[List[str], float, Dict, Optional[List]]
    """
    # Check cache first
    #if cached:
    #    cached_result = session.query(SplitCache).filter_by(input=text_to_split).first()
    #    if cached_result:
    #        cached_splits = ast.literal_eval(cached_result.splitted_text)
    #        print(f"Retrieved from cache: {cached_splits}")
            
            # Even for cached results, we'll score them to ensure best split
    #        if isinstance(cached_splits, list) and isinstance(cached_splits[0], list):
    #            splits_to_score = cached_splits
    #        else:
    #            splits_to_score = [cached_splits]
            
    #        ranked_splits = scorer.rank_splits(splits_to_score)
    #        best_split, best_score, subscores = ranked_splits[0]
    #        
    #        if detailed_output:
    #            return best_split, best_score, subscores, ranked_splits
    #        return best_split

    try:
        # Get all possible splits
        splits = parser.split(text_to_split, limit=attempts)
        
        # Handle None result
        if splits is None:
            simple_split = text_to_split.split()
            if detailed_output:
                ## put a check here to avoid error if missing
                if simple_split:
                    score, subscores = scorer.score_split(simple_split, text_to_split)
                    return simple_split, score, subscores, None
            return simple_split

        # Process splits based on attempts
        if attempts == 1:
            splits = [ast.literal_eval(f'{next(splits)}')]
        else:
            splits = [ast.literal_eval(f'{split}') for split in splits]

        # Score all splits
        #print("Splits", splits)
        ranked_splits = scorer.rank_splits(text_to_split, splits)  # Pass original text
        best_split, best_score, subscores = ranked_splits[0]
        
        # Cache the result if needed
        #if cached:
        #    new_cache_entry = SplitCache(
        #        input=text_to_split, 
        #        splitted_text=str([split for split, _, _ in ranked_splits])
        #    )
        #    session.add(new_cache_entry)
        #    session.commit()
            #print(f"Added to cache: {best_split}")

        if detailed_output:
            return best_split, best_score, subscores, ranked_splits
        return best_split

    except Exception as e:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {e}")
        simple_split = text_to_split.split()
        
        if detailed_output:
            if simple_split:
                score, subscores = scorer.score_split(simple_split, text_to_split)
                return simple_split, score, subscores, None
        return simple_split