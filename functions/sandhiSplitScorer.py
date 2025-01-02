# utils/sandhi_scorer.py
from typing import List, Dict, Tuple
import re
from utils.lexicalResources import (
    VOWEL_SANDHI_INITIALS,
    SANDHI_VARIATIONS_IAST,
    UPASARGAS_WEIGHTS,
    INDECLINABLES
)

class SandhiSplitScorer:
    def __init__(self):
        self.upasargas = UPASARGAS_WEIGHTS
        self.indeclinables = INDECLINABLES
    
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
    
    def calculate_sandhi_score(self, split: List[str], original_text: str) -> float:
        """
        Calculate sandhi score with correct checking of sandhi variations.
        For n splits, we check (n-1) boundaries for valid patterns.
        """
        if len(split) <= 1:
            return 0

        num_boundaries = len(split) - 1
        points_per_boundary = 1.0 / num_boundaries
        base_score = 1.0
        valid_patterns = 0

        valid_unchanged_consonants = {'k', 'p', 't', 'm', 'n', 'ṅ', 'ñ', 'ṇ', 'ś', 'ṣ', 's'}

        for i in range(num_boundaries):
            word1, word2 = split[i], split[i + 1]
            found_valid_pattern = False

            # Vowel sandhi check
            if word1[-1] in VOWEL_SANDHI_INITIALS:
                expected_initials = VOWEL_SANDHI_INITIALS[word1[-1]]
                if any(word2.startswith(init) for init in expected_initials):
                    valid_patterns += 1
                    found_valid_pattern = True

            # Consonant sandhi check - note the reversed logic here
            if not found_valid_pattern and len(word1) > 0 and len(word2) > 0:
                final_char = word1[-1]
                initial_char = word2[0]
                
                # First check if the initial character is in SANDHI_VARIATIONS
                if initial_char in SANDHI_VARIATIONS_IAST:
                    # Then see if the final character could have transformed into it
                    if final_char in SANDHI_VARIATIONS_IAST[initial_char]:
                        valid_patterns += 1
                        found_valid_pattern = True
                
                # Check for valid unchanged consonant sequences
                elif (final_char in valid_unchanged_consonants and 
                    initial_char in valid_unchanged_consonants):
                    valid_patterns += 1
                    found_valid_pattern = True

            # Penalize invalid patterns
            if not found_valid_pattern:
                if (word1[-1] in 'aāiīuūṛṝ' and word2[0] in 'aāiīuūṛṝ'):
                    base_score *= 0.6  # Heavier penalty for invalid vowel sequences
                else:
                    base_score *= 0.8  # Lesser penalty for other invalid patterns

        pattern_ratio = valid_patterns / num_boundaries
        final_score = base_score * pattern_ratio
        return final_score * 0.2
    
    def score_split(self, original_text: str, split: List[str]) -> Tuple[float, Dict[str, float]]:
        scores = {}
        
        # 1. Length scoring - now considers original text length
        length_score = self.calculate_length_score(original_text, split)

        scores['length'] = length_score   
        
        scores['morphology'] = self.calculate_morphology_score(split)
        
        scores['sandhi'] = self.calculate_sandhi_score(split, original_text)
        
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