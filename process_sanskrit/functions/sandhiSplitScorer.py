# utils/sandhi_scorer.py
from typing import List, Dict, Tuple
import re
from process_sanskrit.utils.lexicalResources import (
    VOWEL_SANDHI_INITIALS,
    SANDHI_VARIATIONS_IAST,
    UPASARGAS_WEIGHTS,
    INDECLINABLES
)
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES

class SandhiSplitScorer:
    def __init__(self):
        self.upasargas = UPASARGAS_WEIGHTS
        self.indeclinables = INDECLINABLES


    def words_exist_in_dictionary(self, split: List[str]) -> bool:
        """
        Check if all words in the split exist in DICTIONARY_REFERENCES.
        Returns False if any word is not found.
        """
        for word in split:
            if word not in DICTIONARY_REFERENCES:
                return False
        return True
    
    def calculate_length_score(self, original_text: str, split: List[str]) -> float:
        """
        Calculate length score with stronger preference for fewer splits
        """
        text_length = len(original_text)
        num_splits = len(split)
        
        # Start with perfect score
        base_score = 1.0
        
        # Calculate splits ratio but with higher expectation for characters per split
        splits_ratio = num_splits / (text_length / 8)  # Changed from 6 to 8
        
        # Apply penalty even when ratio <= 1, but more severely above 1
        if splits_ratio > 1:
            base_score *= (1 / (splits_ratio ** 1.3))  # Increased power from 2 to 2.5
        else:
            base_score *= (1 / (splits_ratio ** 1.2))  # Add mild penalty even below ratio 1
        
        return base_score * 0.5
    
    def calculate_morphology_score(self, split: List[str]) -> float:
        """
        Calculate morphology score with recognition of Sanskrit indeclinables and affixes.
        The maximum score of 0.3 is distributed among the words - so for example:
        - Single word compound: that word can get up to 0.3
        - Two word compound: each word can get up to 0.15
        - Three word compound: each word can get up to 0.1
        And so on.
        """
        # Calculate points available per word
        points_per_word = 0.3 / len(split)
        
        morphology_score = 0
        for word in split:
            word_score = 0
            
            # Regular length-based scoring, scaled by points available
            if len(word) >= 6:
                word_score += points_per_word * 0.7  # 70% of available points
            elif len(word) >= 4:
                word_score += points_per_word * 0.4  # 40% of available points
                
            # Additional scaled points for recognized elements
            if word in self.indeclinables:
                word_score += points_per_word * 0.7
            elif word in self.upasargas:
                word_score += points_per_word * 0.4
                
            # Scaled penalty for unrecognized very short words
            if len(word) <= 2 and word not in self.indeclinables and word not in self.upasargas:
                word_score -= points_per_word
                
            # Reward Sanskrit endings with scaled points
            if re.search(r'(ana|ita|aka|in|tva|tā)$', word):
                word_score += points_per_word * 0.7
                
            morphology_score += word_score
        
        return max(0, min(morphology_score, 0.3))
    
    
    def score_split(self, original_text: str, split: List[str]) -> Tuple[float, Dict[str, float]]:

        scores = {}
        
        # 1. Length scoring - now considers original text length
        length_score = self.calculate_length_score(original_text, split)

        scores['length'] = length_score   
        
        scores['morphology'] = self.calculate_morphology_score(split)
        
        scores['sandhi'] = 0
        
        total_score = sum(scores.values())
        return total_score, scores

    def rank_splits(self, original_text: str, splits: List[List[str]]) -> List[Tuple[List[str], float, Dict[str, float]]]:
        scored_splits = []
        for split in splits:
            score, subscores = self.score_split(original_text, split)
            scored_splits.append((split, score, subscores))
        
        return sorted(scored_splits, key=lambda x: x[1], reverse=True)

    

# Create a global scorer instance

scorer = SandhiSplitScorer()

