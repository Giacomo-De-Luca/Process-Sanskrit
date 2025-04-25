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
        
        scores['sandhi'] = 0
        
        total_score = sum(scores.values())
        return total_score, scores

    def rank_splits(self, original_text: str, splits: List[List[str]]) -> List[Tuple[List[str], float, Dict[str, float]]]:
        scored_splits = []
        for split in splits:
            score, subscores = self.score_split(original_text, split)
            scored_splits.append((split, score, subscores))
        
        return sorted(scored_splits, key=lambda x: x[1], reverse=True)


class SandhiSplitScorer:
    def __init__(self):
        # Previous dictionaries remain the same
        self.upasargas = {
            'ā': 0.1, 'ati': 0.2, 'adhi': 0.2, 'anu': 0.1,
            'apa': 0.2, 'api': 0.1, 'abhi': 0.2, 'ava': 0.2,
            'ud': 0.1, 'upa': 0.2, 'dur': 0.2, 'ni': 0.1,
            'nir': 0.2, 'nis': 0.2, 'parā': 0.2, 'pari': 0.2,
            'pra': 0.2, 'prati': 0.2, 'vi': 0.1, 'sam': 0.2,
            'su': 0.1
        }
        self.indeclinables = {
            'iva', 'eva', 'ca', 'vā', 'hi', 'tu', 'api',
            'iti', 'yathā', 'tathā', 'yatra', 'tatra'
        }
    

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
        Calculate sandhi score by checking if the splits follow valid sandhi rules.
        Uses existing VOWEL_SANDHI_INITIALS and SANDHI_VARIATIONS dictionaries.
        
        For example, in 'tajjñānam' → ['tad', 'jñānam']:
        - Recognizes 'd' → 'j' as valid sandhi (SANDHI_VARIATIONS['d'] includes 'j')
        - Validates the transformation at the boundary
        """
        sandhi_score = 0  # Start with base score
        
        for i in range(len(split) - 1):
            word1, word2 = split[i], split[i+1]
            
            # Check for valid sandhi transformations at word boundaries
            valid_sandhi = False
            
            # Case 1: Vowel Sandhi
            if word1[-1] in VOWEL_SANDHI_INITIALS:
                expected_initials = VOWEL_SANDHI_INITIALS[word1[-1]]
                if any(word2.startswith(init) for init in expected_initials):
                    sandhi_score += 0.1  # Reward valid vowel sandhi
                    valid_sandhi = True
            
            # Case 2: Consonant Sandhi
            if not valid_sandhi and len(word1) > 0 and len(word2) > 0:
                final_char = word1[-1]
                initial_char = word2[0]

                print("final_char, initial_char", final_char, initial_char)
                
                # Check if this is a valid consonant transformation
                if final_char in SANDHI_VARIATIONS_IAST:
                    valid_variants = SANDHI_VARIATIONS_IAST[final_char]
                    if initial_char in valid_variants:
                        sandhi_score += 0.15  # Higher reward for consonant sandhi
                        valid_sandhi = True
                
                # Special case for common Sanskrit transformations
                # Like 't/d' + 'j' → 'j' (as in tad+jñānam → tajjñānam)
                if (final_char in ['t', 'd'] and initial_char == 'j'):
                    sandhi_score += 0.2  # Highest reward for complex sandhi
                    valid_sandhi = True
            
            # Penalize invalid combinations only if no valid sandhi was found
            if not valid_sandhi:
                # Check for unlikely vowel sequences
                if (word1[-1] in 'aāiīuūṛṝ' and word2[0] in 'aāiīuūṛṝ'):
                    sandhi_score -= 0.1
                
                # Penalize breaking natural compounds
                if (len(word1) <= 2 and i > 0) or (word2 == 'ni' and i == len(split) - 2):
                    sandhi_score -= 0.1
        
        return max(0, min(sandhi_score, 0.3)) * 0.2  # Scale to [0, 0.2] range
    
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

