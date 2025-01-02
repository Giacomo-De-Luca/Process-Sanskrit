import indic_transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import ast
from utils.detectTransliteration import detect
import logging
import json
import sqlite3
import re
import pandas as pd
from sanskrit_parser import Parser
from tabulate import tabulate
import regex
from itertools import groupby
import unicodedata
from typing import List, Dict, Any
from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass



variableSandhiSLP1 = {
    'k': ['t', 'c', 'S'],        # From t/c/S in word-final position
    'w': ['t', 'S', 'z'],        # From t/S/z in word-final position
    't': ['d'],                   # From d in word-final position
    'n': ['t', 'm', 'M'],        # From t/m/aM in word-final position
    'p': ['t', 'b'],             # From t/b in word-final position
    'q': ['t', 'd'],             # From t/d in word-final position
    's': ['S', 'z', 'H'],        # From S/z/H in word-final position
    'S': ['k', 'H'],             # From k/H in word-final position
    'z': ['k', 'H', 't'],        # From k/H/t in word-final position
    'r': ['H', 's'],             # From H/s in word-final position
    'o': ['aH', 'as', 'O'],     # From as/aH/O in word-final position
    'j': ['t'], 
    'M': ['m'], 

}

sanskritFixedSandhiMapSLP1 = {
    'y': 'i',          # 'I' out for testing # y comes from i or I before vowels (like devI + atra → devyatra)
    'r': 'H',          # r comes from visarga before voiced sounds and some vowels (like punaH + gacCati → punargacCati)
    #'N': 'n',          # N comes from n before velars (k, K, g, G) (like tAn + karoti → tANkaroti)
    #'Y': 'n',          # Y comes from n before palatals (c, C, j, J) (like tAn + carati → tAYcarati)
    #'R': 'n',          # R comes from n before retroflexes (w, W, q, Q) (like tAn + wIkate → tARwIkate)
    'v': 'u',          # 'U' out for testing  # v comes from u or U before vowels (like guru + atra → gurvatra)
    'd': 't',          # d comes from t before voiced consonants (like tat + dAnam → taddAnam)
    'b': 'p',          # b comes from p before voiced consonants (like ap + BiH → abBiH)
    'g': 'k',          # g comes from k before voiced consonants (like vAk + devi → vAgdevi)
}





__version__ = "0.2"
def print_version():
    print(f"Version: {__version__}")


import os
#from dotenv import load_dotenv

## If heroku postgres
#load_dotenv()
#DATABASE_URL = os.environ['DATABASE_URL']

#if local postgres

#DATABASE_URL = "postgresql+psycopg2://postgres:again@localhost:5432/sanskritmagicdb"
#DATABASE_URL = os.getenv("DATABASE_URL")
#DATABASE_URL = os.getenv("postgres://u5o7c326q19pvp:pdb08c4f74fd1c63df61a559fc3d7a261ac3c65a24df6cfd06fbb2ad511143f0d@c3cj4hehegopde.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d61ijmaljbh829")

#if DATABASE_URL.startswith("postgres://"):
    #DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

#if using SQLite

DATABASE_URL = "sqlite:///resources/merged_formdb.sqlite"

logging.basicConfig(level=logging.CRITICAL)

##to get all the available schemes
##indic_transliteration.sanscript.SCHEMES.keys()

def transliterateIASTSLP1(text):
    return transliterate(text, sanscript.IAST, sanscript.SLP1)   
    
def transliterateSLP1IAST(text):
    return transliterate(text, sanscript.SLP1, sanscript.IAST)   

def transliterateSLP1HK(text):
    return transliterate(text, sanscript.SLP1, sanscript.HK)   

def transliterateDEVSLP1(text):
    return transliterate(text, sanscript.DEVANAGARI, sanscript.SLP1)
        
def anythingToSLP1(text):
    detected_scheme_str = detect(text).upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    return sanscript.transliterate(text, detected_scheme, sanscript.SLP1)
def anythingToIAST(text):
    detected_scheme_str = detect(text).upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    return sanscript.transliterate(text, detected_scheme, sanscript.IAST)
def anythingToHK(text):
    detected_scheme_str = detect(text).upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    return sanscript.transliterate(text, detected_scheme, sanscript.HK)

def transliterateAnything(text, transliteration_scheme):
    detected_scheme_str = detect(text).upper()
    transliteration_scheme_str = transliteration_scheme.upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    output_scheme = getattr(sanscript, transliteration_scheme_str)
    return sanscript.transliterate(text, detected_scheme, output_scheme)

##qui la lista degli encoding è lowercase

parser = Parser(output_encoding='iast')

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
# Create a session
session = Session()

Base = declarative_base()

# Define the split_cache model
class SplitCache(Base):
    __tablename__ = 'split_cache'
    
    input = Column(String, primary_key=True)
    splitted_text = Column(String)

# Create the table if it doesn't exist
Base.metadata.create_all(engine)



def sandhi_splitter(text_to_split, cached=True, attempts=1):
    """
    Splits the given text using a sandhi splitter parser.
    Checks if the result is already cached before performing the split.

    Parameters:
    - text_to_split (str): The text to split.

    Returns:
    - list: A list of split parts of the text.
    """

    #text_to_split = anythingToSLP1(text_to_split).strip()

    # Check if the result is already cached
    if cached == True:
        cached_result = session.query(SplitCache).filter_by(input=text_to_split).first()
    
    if cached == True: 
        if cached_result:
            # Retrieve and return the cached result if it exists
            splitted_text = ast.literal_eval(cached_result.splitted_text)
            print(f"Retrieved from cache: {splitted_text}")
            return splitted_text

    # If not cached, perform the split
    try:
        splits = parser.split(text_to_split, limit=attempts)

        #if split is none, default to split by space
        if splits is None:
            return text_to_split.split()
        
        if attempts == 1: 

            for split in splits:
                splitted_text = f'{split}'
            splitted_text = ast.literal_eval(splitted_text)

        if attempts > 1: 

            splitted_text = []
            for split in splits:
                string_split =  f'{split}'
                splitted_text.append(ast.literal_eval(string_split))



        print(f"Splitted text: {splitted_text}")

        # Store the split result in cache as a list
        if cached == True: 
            new_cache_entry = SplitCache(input=text_to_split, splitted_text=str(splitted_text))
            session.add(new_cache_entry)
            session.commit()
            print(f"Added to cache: {splitted_text}")

        return splitted_text

    except Exception as e:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {e}")
        return text_to_split.split()




@dataclass
class SplitResult:
    """Class to hold the result of a sandhi split with scoring information"""
    split: List[str]
    score: float
    subscores: dict
    all_splits: Optional[List[Tuple[List[str], float, dict]]] = None

def enhanced_sandhi_splitter(
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
    if cached:
        cached_result = session.query(SplitCache).filter_by(input=text_to_split).first()
        if cached_result:
            cached_splits = ast.literal_eval(cached_result.splitted_text)
            print(f"Retrieved from cache: {cached_splits}")
            
            # Even for cached results, we'll score them to ensure best split
            if isinstance(cached_splits, list) and isinstance(cached_splits[0], list):
                splits_to_score = cached_splits
            else:
                splits_to_score = [cached_splits]
            
            ranked_splits = scorer.rank_splits(splits_to_score)
            best_split, best_score, subscores = ranked_splits[0]
            
            if detailed_output:
                return best_split, best_score, subscores, ranked_splits
            return best_split

    try:
        # Get all possible splits
        splits = parser.split(text_to_split, limit=attempts)
        
        # Handle None result
        if splits is None:
            simple_split = text_to_split.split()
            if detailed_output:
                score, subscores = scorer.score_split(simple_split)
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
        if cached:
            new_cache_entry = SplitCache(
                input=text_to_split, 
                splitted_text=str([split for split, _, _ in ranked_splits])
            )
            session.add(new_cache_entry)
            session.commit()
            #print(f"Added to cache: {best_split}")

        if detailed_output:
            return best_split, best_score, subscores, ranked_splits
        return best_split

    except Exception as e:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {e}")
        simple_split = text_to_split.split()
        
        if detailed_output:
            score, subscores = scorer.score_split(simple_split)
            return simple_split, score, subscores, None
        return simple_split
    

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
        
        scores['sandhi'] = self.calculate_sandhi_score(split, original_text)
        
        total_score = sum(scores.values())
        return total_score, scores

    def rank_splits(self, original_text: str, splits: List[List[str]]) -> List[Tuple[List[str], float, Dict[str, float]]]:
        scored_splits = []
        for split in splits:
            score, subscores = self.score_split(original_text, split)
            scored_splits.append((split, score, subscores))
        
        return sorted(scored_splits, key=lambda x: x[1], reverse=True)
    

scorer = SandhiSplitScorer()


stopwords = pd.read_csv('resources/stopwords.csv')

stopwords_as_list = stopwords['stopword'].tolist()

def remove_stopwords_list(text_list):
    return [word for word in text_list if word not in stopwords_as_list]

def remove_stopwords_string(text):
    text = text.replace('.', '')  # Remove periods
    text_list = text.split()  # Split the string into words
    return ' '.join(word for word in text_list if word not in stopwords_as_list)


with open('resources/MWKeysOnly.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeys = json.load(f)

with open('resources/MWKeysOnlySLP1.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeysSLP1 = json.load(f)


##given a name finds the root


def SQLite_find_name(name):

    outcome = []    

    def query1(word):

        session = Session()
        try:
            query_builder = text("SELECT * FROM lgtab2 WHERE key = :word")
            results = session.execute(query_builder, {'word': word}).fetchall()
        except Exception as error:
            print("Error while querying PostgreSQL:", error)
            results = []
        finally:
            session.close()
        return results

    results = query1(name)
    
    if not results:  # If query1 didn't find any results
        if name[-1] == 'M':
            name = name[:-1] + 'm'
            results = query1(name)
    
    for inflected_form, type, root_form in results: 
        if not root_form:  # If root_form is None or empty
            return  # End the function

        def query2(root_form: str, type: str):

            session = Session()
            try:
                query_builder2 = text("SELECT * FROM lgtab1 WHERE stem = :root_form and model = :type ")
                results = session.execute(query_builder2, {'root_form': root_form, 'type': type}).fetchall()
            except Exception as error:
                print("Error while querying PostgreSQL:", error)
                results = []
            finally:
                session.close()
            return results
        
        result = query2(root_form, type)
        word_refs = re.findall(r",([a-zA-Z]+)",result[0][2])[0]
        inflection_tuple = result[0][3]  # Get the first element of the first tuple
        inflection_words = inflection_tuple.split(':') 

        ##transliterate back the result to IAST for readability
        inflection_wordsIAST = [transliterateSLP1IAST(word) for word in inflection_words]
        query_transliterateIAST = transliterateSLP1IAST(name)

        ##make Inflection Table
        indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]
        rowtitles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
        coltitles = ["Sg", "Du", "Pl"]

        if indices:
            row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
        else: 
            row_col_names = None
        outcome.append([word_refs, type, row_col_names, inflection_wordsIAST, transliterateSLP1IAST(name)])

    return outcome



def SQLite_find_verb(verb):
    
    root_form = None

    def query1(verb):

        session = Session()
        try:
            query_builder = text("SELECT * FROM vlgtab2 WHERE key = :verb")
            results = session.execute(query_builder, {'verb': verb}).fetchall()
        except Exception as error:
            print("Error while querying PostgreSQL:", error)
            results = []
        finally:
            session.close()
        return results

    result = query1(verb)
    
    for inflected_form, type, root_form in result:

        if not root_form:  # If root_form is None or empty
            return  # End the function
        type_var = type

        def query2(root_form: str, type: str):

            session = Session()
            try:
                query_builder2 = text("SELECT * FROM vlgtab1 WHERE stem = :root_form and model = :type")
                results = session.execute(query_builder2, {'root_form': root_form, 'type': type}).fetchall()
            except Exception as error:
                print("Error while querying PostgreSQL:", error)
                results = []
            finally:
                session.close()
            return results
        
        result = query2(root_form, type)
    
    selected_tuple = None

    # Iterate over the result list
    for model, stem, refs, data in result:
        if model == type_var:  # If the model matches type_var
            ref_word = re.search(",([a-zA-Z]+)", refs).group(1)
            if stem != ref_word:
                stem= ref_word
                #print("ref_word, stem", ref_word, stem)
                selected_tuple = (model, stem, refs, data)  # Get the entire tuple
                break  # Exit the loop
            selected_tuple = (model, stem, refs, data)  # Get the entire tuple
            break  # Exit the loop

    if selected_tuple is None:
        #print("No matching model found in result")
        return
    

    # Now you can use selected_tuple
    inflection_tuple = selected_tuple[3]  # Get the 'data' element of the tuple
    inflection_words = inflection_tuple.split(':') 
    
    ##transliterate back the result to IAST for readability
    
    inflection_wordsIAST = [transliterateSLP1IAST(word) for word in inflection_words]
    query_transliterateIAST = transliterateSLP1IAST(verb)
    
    ##make Inflection Table
    
    indices = [i for i, x in enumerate(inflection_wordsIAST) if x == query_transliterateIAST]

    # Define row and column titles
    rowtitles = ["First", "Second", "Third"]
    coltitles = ["Sg", "Du", "Pl"]


    if indices:
        row_col_names = [(rowtitles[i//3], coltitles[i%3]) for i in indices]
    else:
        row_col_names = None
        
    return [[stem, type_var, row_col_names, inflection_wordsIAST, transliterateSLP1IAST(verb)]]


## also map to the type.
# Read the Excel file into a DataFrame
type_map = pd.read_excel('resources/type_map.xlsx')

def root_any_word(word, attempted_words=None):
    if attempted_words is None:
        attempted_words = set()

    # If the word has already been attempted, return None to avoid infinite loop
    if word in attempted_words:
        return None

    # Add the current word to the set of attempted words
    attempted_words.add(word)

    result_roots_name = SQLite_find_name(word)  
    result_roots_verb = SQLite_find_verb(word) 

    if result_roots_name:
        if result_roots_verb:
            result_roots = result_roots_name + result_roots_verb
        else:
            result_roots = result_roots_name
    else:
        result_roots = result_roots_verb

    if result_roots:
        for i in range(len(result_roots)):
            result = result_roots[i]
            # Get the second member of the list
            abbr = result[1]
            # Find the matching value in the 'abbr' column
            match = type_map[type_map['abbr'] == abbr]
            
            if not match.empty:
                description = match['description'].values[0]
                result[1] = description
                result_roots[i] = result
        return result_roots

    # If no result is found, try replacements based on variableSandhiSLP1
    if word[-1] in variableSandhiSLP1:
        for replacement in variableSandhiSLP1[word[-1]]:
            tentative = word[:-1] + replacement
            attempt = root_any_word(tentative, attempted_words)
            if attempt is not None:
                return attempt
    
    samMap = {
            'sam' : 'saM',
            'saM' : 'sam',
            'saN' : 'saM',
            'san' : 'saM',
            'saY' : 'saM',
        }

        ##different spellings for sam, - it is so common that it deserves its own rule
    if word[0:3] in samMap:
            print("tentative", word)
            tentative = samMap[word[0:3]] + word[3:]
            attempt = root_any_word(tentative, attempted_words)
            print("attempt", attempt)
            if attempt is not None:
                result = attempt
                return attempt

    return None





def root_any_list(text_list):
    roots = []
    for word in text_list:
        root_found = root_any_word(word)
        if root_found is not None:
            roots.append(root_found)
    
    # Transliterate the first element of each tuple
    for i in range(len(roots)):
        roots[i] = (transliterateSLP1IAST(roots[i][0].replace('-', '')),) + roots[i][1:]
    
    return roots


        ##if the dictionary approach fails, try the iterative approach:


def inflect(splitted_text):
    roots = []
    prefixes = ['sva', 'anu', 'sam', 'pra', 'upa', 'vi', 'nis', 'aBi', 'ni', 'pari', 'prati', 'parA', 'ava', 'aDi', 'api', 'ati', 'ud', 'dvi', 'su', 'dur', 'duH']  # Add more prefixes as needed
    i = 0
    while i < len(splitted_text):
        word = splitted_text[i]
        if word in prefixes and i + 1 < len(splitted_text):
            next_word = splitted_text[i + 1]
            if word == 'sam':
                combined_words = ['sam' + next_word, 'saM' + next_word]
            elif word == 'vi':
                combined_words = ['vi' + next_word, 'vy' + next_word]
            else:
                combined_words = [word + next_word]

            rooted = None
            for combined_word in combined_words:
                rooted = root_any_word(combined_word)
                if rooted is not None:
                    break  # Exit loop if a valid root is found

            if rooted is not None:
                roots.extend(rooted)
                i += 2  # Skip next word since it's part of the combined word
                continue
            else:
                rooted_word = root_any_word(word)
                if rooted_word is not None:
                    roots.extend(rooted_word)
                else:
                    compound_try = root_compounds(word)
                    if compound_try is not None:
                        roots.extend(compound_try)
                    else:
                        roots.append(word)
                i += 1  # Move to next word
        else:
            rooted = root_any_word(word)
            if rooted is not None:
                roots.extend(rooted)
            else:
                compound_try = root_compounds(word)
                if compound_try is not None:
                    roots.extend(compound_try)
                else:
                    roots.append(word)
            i += 1

    # Transliterate roots
    for j in range(len(roots)):
        if isinstance(roots[j], list):
            roots[j][0] = transliterateSLP1IAST(roots[j][0].replace('-', ''))
        else:
            roots[j] = transliterateSLP1IAST(roots[j].replace('-', ''))
    return roots


## bug with process("nīlotpalapatrāyatākṣī")    




# Dictionary mapping final vowels to possible initial vowels in SLP1 notation
VOWEL_SANDHI_INITIALS = {
    # When a word ends in 'A', the next word might have lost initial 'a' or 'A'
    'A': ['a', 'A'],
    
    # For final 'a', check for lost initial 'i'/'I' (e) or 'u'/'U' (o)
    #'a': ['i', 'I', 'u', 'U'],
    
    # For final 'i'/'I', the next word might have lost initial 'i'/'I'
    #'i': ['i',],
    'I': ['i', 'I'],
    
    # For final 'u'/'U', the next word might have lost initial 'u'/'U'
    #'u': ['u', 'U'],
    'U': ['u', 'U'],
    
    # For final 'e', check for lost initial 'a'/'A'
    'e': ['i'],
    
    # For final 'o', check for lost initial 'a'/'A'
    'o': ['u'],


}

# New dictionary for sandhi variations in final letters
SANDHI_VARIATIONS = {
    # Vowel variations
    'A': ['a', 'A'],
    'I': ['i', 'I'],
    'U': ['u', 'U'],
    'C': ['S'],
    # Visarga variations
    'H': ['s', 'r', 'H'],
    'o': ['a', 'A', 'o'],
    'e': ['a', 'A', 'e'],
    'E': ['e', 'E'],
    'O': ['o', 'O'],
    
    # Common consonant variations
    'n': ['m', 'M', 'n'],
    't': ['d', 't'],
    'd': ['t', 'd'],
    
    # Nasal variations
    'M': ['m', 'n', 'N', 'Y', 'R'],
    
    # Other common variations
    'c': ['k', 'd'],
    'j': ['k', 'g', 'j', 'd', 't'],
    'z': ['s', 'z', 'S'],
    'y': ['i', 'y'],
    'v': ['v', 'u'],

    'C': ['t', 'S'], # P0or cases like tacCabdaH

'    N': ['n', 'N'], # P0or cases like saNgacCati

    'M': ['m', 'n'],
    
}

SANDHI_VARIATIONS_IAST = {
    # vowel variations

'ā': ['a', 'ā'],

'ī': ['i', 'ī'],

'ū': ['u', 'ū'],

'ch': ['ś'],

# Visarga variations

'ḥ': ['s', 'r', 'ḥ'],

'o': ['a', 'ā', 'o'],

'e': ['a', 'ā', 'e'],

'ai': ['e', 'ai'],

'au': ['o', 'au'],

# chommon consonant variations

'n': ['m', 'ṃ', 'n'],

't': ['d', 't'],

'd': ['t', 'd'],

# ṅasal variations

'ṃ': ['m', 'n', 'ṅ', 'ñ', 'ṇ'],

# auther common variations

'c': ['k', 'd'],

'j': ['k', 'g', 'j', 'd', 't'],

'ṣ': ['s', 'ṣ', 'ś'],

'y': ['i', 'y'],

'v': ['v', 'u'],

'ch': ['t', 'ś'],  # For cases like tacchabdaḥ

'ṅ': ['n', 'ṅ'],   # For cases like saṅgacchati

'ṃ': ['m', 'n'], 

}




SANSKRIT_PREFIXES = {
    'sam': 'together, completely',
    'anu': 'along, after',
    'aBi': 'towards, into',
    'ati': 'beyond, over',
    'aDi': 'over, upon',
    'apa': 'away, off',
    'api': 'unto, close',
    'ava': 'down, off',
    'A': 'near to, completely',
    'ud': 'up, upwards',
    'upa': 'towards, near',
    'nis': 'out, away',
    'parA': 'away, back',
    'pari': 'around, about',
    'pra': 'forward, forth',
    'prati': 'towards, back',
    'vi': 'apart, away',
    'ut': 'up, upwards',  # Variant of ud- before certain consonants
    'ni': 'down, into'
}

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
    if word in mwdictionaryKeysSLP1:
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
            if remainder in mwdictionaryKeysSLP1:
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
        if temp_word in mwdictionaryKeysSLP1:
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
                if test_word in mwdictionaryKeysSLP1:
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


            if remaining.startswith('C'):
                test_word = 'S' + remaining[1:]
                if debug:
                    print(f"Trying with S instead of C: {test_word}")
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
        return transliterateSLP1IAST(root_result)
    # If it's a list, the last element is usually the base form
    # Lists from root_compounds typically end with the base form
    return transliterateSLP1IAST(root_result[0])



def hybrid_sandhi_splitter(
    text_to_split: str,
    cached: bool = False,
    attempts: int = 10,
    detailed_output: bool = False,
    score_threshold: float = 0.535
) -> Union[List[str], Tuple[List[str], float, Dict, List]]:
    """
    Enhanced sandhi splitter that combines statistical splitting with root compound analysis.
    Processes complex root analysis output into scoreable word lists.
    
    Parameters:
    - text_to_split: Text to split
    - cached: Whether to use caching
    - attempts: Number of splitting attempts for statistical method
    - detailed_output: If True, returns additional scoring information
    - score_threshold: Minimum score to accept statistical split
    """
    # First try our enhanced statistical splitter
    if detailed_output:
        stat_split, stat_score, stat_subscores, all_splits = enhanced_sandhi_splitter(
            text_to_split, cached, attempts, detailed_output=True
        )
        if len(stat_split) == 1:
            stat_score = 0 
    else:
        stat_split = enhanced_sandhi_splitter(
            text_to_split, cached, attempts, detailed_output=False
        )
        # Get the score for comparison
        if len(stat_split) == 1:
            stat_score = 0 
        else: 
            stat_score, stat_subscores = scorer.score_split(text_to_split, stat_split)


    # If score is good enough, return statistical result
    if stat_score >= score_threshold:
        print("stat_score", stat_score)
        print("stat_split", stat_split)
        if detailed_output:
            return stat_split, stat_score, stat_subscores, all_splits
        return stat_split

    # If score is too low, try root compound analysis
    try:
        print("text_to_split", text_to_split)
        root_analysis = root_compounds(transliterateIASTSLP1(text_to_split), inflection=False)
        print("root_analysis", root_analysis)
        if root_analysis:
            # Process the root analysis results into a simple word list
            root_split = [process_root_result(item) for item in root_analysis]
            root_split = [x for i, x in enumerate(root_split) if i == 0 or x != root_split[i-1]]
            print("root_split", root_split)
            
            # Score the processed root split
            root_score, root_subscores = scorer.score_split(text_to_split, root_split)
            print("root_score", root_score)
            print("root_subscores", root_subscores)

            # Compare scores and choose the better result
            if root_score > stat_score:
                if detailed_output:
                    # Add root split to all_splits for reference
                    all_splits = [(root_split, root_score, root_subscores)] + (all_splits if all_splits else [])
                    return root_split, root_score, root_subscores, all_splits
                return root_split

    except Exception as e:
        print(f"Root compound analysis failed: {str(e)}")

    # Fall back to statistical split if root analysis fails or scores lower
    print("stat_split", stat_split)
    if detailed_output:
        return stat_split, stat_score, stat_subscores, all_splits
    return stat_split






def process(text, *dict_names, max_length=100, debug=False, root_only=False, ):


    ## if the text is too long, we try to trim it to the last whitespace
    if len(text) > max_length:
        last_space_index = text[:max_length].rfind(' ')
        if last_space_index == -1:
            text = text[:max_length]
        else:
            # Trim up to the last whitespace
            text = text[:last_space_index]

    ## jj occours only in sandhi and we know it should be split into two words. 
    if 'jj' in text:
        text = text.replace('jj', 'j j')

    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails

    if ' ' not in text:

        ## if the text end with a *, remove it and try to find the word in the dictionary for exact match
        if text[-1] == '*':
            voc_entry = get_voc_entry([anythingToIAST(text[:-1])], *dict_names)
            if voc_entry is not None:
                return voc_entry
            else:
                process(text[:-1])

        ## if in the text there is a _, use wildcard search directly

        if '_' in text or '%' in text:
            print("wildcard search", anythingToIAST)
            voc_entry = get_voc_entry([anythingToIAST(text)], *dict_names)
            if voc_entry is not None:
                return voc_entry
            else:
                process(text)

        ## if the text is already sandhi split keep it sandhi split
        if "-" in text or "+" in text:
            # Use re.split to split by either "-" or "+"
            word_list = re.split(r'[-+]', text)
            processed_results = []
            for word in word_list:
                result = process(word)
                processed_results = processed_results + result
            return processed_results
        
        #print("single_word", text)
        transliterated_text = anythingToSLP1(text)     
        #print("transliterated_text", transliterated_text)

        ## remove all non-alphabetic characters
        text = regex.sub('[^\p{L}\']', '', transliterated_text)

        ## do some preliminary cleaning using sandhi rules ## to remove use a map of tests to apply, and a map of replacements v --> u, s-->H, etc
        if text[0] == "'":
            text = 'a' + text[1:]
        
        if text[-1] in sanskritFixedSandhiMapSLP1:
            text = text[:-1] + sanskritFixedSandhiMapSLP1[text[-1]]

        elif text[-1] == 'S':
            text = text[:-1] + 'H'

        #print("text", text)
        if "o'" in text:
            modified_text = re.sub(r"o'", "aH a", text)
            #print("modified_text", modified_text)
            result = process(modified_text)
            return result

        ## if the text is a single word, try to find the word first using the inflection table then if it fails on the dictionary for exact match, the split if it fails
        result = root_any_word(text)

        # Look for prefixes only at the start
        if result is None:
            for prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                if text.startswith(prefix):
                    remainder = text[len(prefix):]
                    attempt = root_any_word(remainder)
                    if attempt is not None:
                        result = [prefix] + attempt
                        break
            ## commented out as it breaks a lot of words like samadhi // it matches too much
            #       else:

                        ### breaks with ps.process("samādhipariṇāmaḥ")
                        #for nested_prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                            #if remainder.startswith(nested_prefix):
                                #nested_remainder = remainder[len(prefix):]
                                #print("nested_remainder", nested_remainder)
                                #nested_attempt = root_any_word(nested_remainder)
                                #if nested_attempt is not None:
                                    #result = [prefix] + [nested_prefix] + nested_attempt
                                    #break

        ## if the word ends with a M, try to correct it for different conventions  and find the word in the dictionary for exact match
        # Check if the word ends with a character that is a key in variableSandhiSLP1
        if result is None and text[-1] in variableSandhiSLP1:
            for replacement in variableSandhiSLP1[text[-1]]:
                tentative = text[:-1] + replacement
                attempt = root_any_word(tentative)
                if attempt is not None:
                    result = attempt
                    break

        ## if the words starts with C, try to find out if it's the sandhied form of a word starting with S
        if result is None and text[0:1] == "C":
            #print("tentative", text)
            tentative = 'S' + text[1:] 
            attempt = root_any_word(tentative)
            #print("attempt", attempt)
            if attempt is not None:
                result = attempt
        
        if result is not None:
            if debug == True: 
                print("Getting some results with no splitting here:", result)

            for i, res in enumerate(result):
                if isinstance(res, str):
                    result[i] = transliterateSLP1IAST(res.replace('-', ''))
                elif isinstance(res, list):
                    if isinstance(res[0], str):
                        res[0] = transliterateSLP1IAST(res[0].replace('-', ''))
            result_vocabulary = get_voc_entry(result, *dict_names)

            if debug == True: 
                print("result_vocabulary", result_vocabulary)

            ## if the word is inside the dictionary, we return the entry directly, since it will be accurate. 
            if isinstance(result_vocabulary, list):
                
                if len(result[0]) > 4 and result[0][0] != result[0][4] and result[0][4] in mwdictionaryKeys:
                    replacement = get_voc_entry([result[0][4]], *dict_names)
                    print("replacement", replacement[0])
                    print("len replacement", len(replacement[0]))
                    if replacement is not None:
                        result_vocabulary.insert(0, replacement[0])

            #print("result_vocabulary", result_vocabulary)
            return clean_results(result_vocabulary, debug=debug, root_only=root_only)
        else:
            query = [transliterateSLP1IAST(text)]
            #print("query", query)
            result_vocabulary = get_voc_entry(query, *dict_names)  
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return clean_results(result_vocabulary, debug=debug, root_only=root_only)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    ## attempt to remove sandhi and tokenise in any case
    if debug == True:
        print("pre_splitted_text", text)
    text = transliterateSLP1IAST(text)
    #print("transliterate to split", text)
    splitted_text = hybrid_sandhi_splitter(text)
    if debug == True:
        print("splitted_text", splitted_text)
    splitted_text = [transliterateIASTSLP1(word) for word in splitted_text]    
    inflections = inflect(splitted_text) 
    inflections_vocabulary = get_voc_entry(inflections, *dict_names)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]       

    return clean_results(inflections_vocabulary, debug=debug, root_only=root_only)


##process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")

filtered_words = ["ca", "na", "eva", "ni", "apya", "ava", "sva"]


def clean_results(list_of_entries, root_only=False, debug=True):

    i = 0
    if debug == True:
        print("it breaks right here:", list_of_entries)

    #print("is it broken here?", list_of_entries[i])

    while i < len(list_of_entries) - 1:  # Subtract 1 to avoid index out of range error
        # Check if the word is in filtered_words
        if list_of_entries[i][0] in filtered_words:
            while i < len(list_of_entries) - 1 and list_of_entries[i + 1][0] == list_of_entries[i][0]:
                del list_of_entries[i + 1]

        if list_of_entries[i][0] == "duḥ" and list_of_entries[i+1][0] == "kha":
            replacement = get_voc_entry(["duḥkha"])
            if replacement is not None:
                list_of_entries[i] = replacement[0]
                del list_of_entries[i + 1]
                if list_of_entries[1+2] == "kha":
                    del list_of_entries[i + 2]  ##it's kha also as well

        if len(list_of_entries[i]) >= 5 and list_of_entries[i][0][-1] == "n" and list_of_entries[i][4] != list_of_entries[i][0]:
            #print("the one not replaced:", list_of_entries[i])
            if list_of_entries[i][4] in mwdictionaryKeys:
                replacement = get_voc_entry([list_of_entries[i][4]])
                if replacement is not None:
                    list_of_entries[i] = replacement[0]
        

        
        # Check if the word is "sam"
        if list_of_entries[i][0] == "sam":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "sa" or list_of_entries[j][0] == "sam"):
                j += 1
            if j < len(list_of_entries):


                                ##non ha senso CHECK IF sam or sam  + list_of_entries[j][0]] are in MW dict
                                ## a quel punto fai voc_entry di quello, e rimpiazza tutte le entry inutili.

                voc_entry = get_voc_entry(["sam" + list_of_entries[j][0]])
                #print("voc_entry", voc_entry)

                ##non ha senso
                
                # Ensure voc_entry is not None and has the expected structure
                if (voc_entry is not None and len(voc_entry) > 0 and 
                    isinstance(voc_entry[0], list) and len(voc_entry[0]) > 2 and 
                    isinstance(voc_entry[0][2], dict) and 'MW' in voc_entry[0][2]):
                    
                    # Check if the first key of the dictionary inside MW matches the condition
                    first_key = next(iter(voc_entry[0][2]['MW']), None)
                    if first_key and voc_entry[0][0] == first_key:
                        print("revised query", ["saṃ" + list_of_entries[j][0]])
                        voc_entry = get_voc_entry("saṃ" + list_of_entries[j][0])
                        print("revise_voc_entry", voc_entry)
        
                if voc_entry is not None:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]
        
        # Check if the word is "anu"
        if list_of_entries[i][0] == "anu":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "anu"):
                j += 1
            if j < len(list_of_entries):
                voc_entry = get_voc_entry(["anu" + list_of_entries[j][0]])
                if voc_entry is not None:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]
        
        # Check if the word is "ava"
        if list_of_entries[i][0] == "ava":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "ava"):
                j += 1
            if j < len(list_of_entries):
                print("testing with:", ["ava" + list_of_entries[j + 1][0]])
                voc_entry = get_voc_entry(["ava" + list_of_entries[j + 1][0]])
                if voc_entry is not None:
                    list_of_entries[i] = [item for sublist in voc_entry for item in sublist]
                    del list_of_entries[i + 1:j + 1]        
        i += 1  
    
    print("list of roots:")

    def roots(list_of_entries, debug=debug):
        roots = []
        for entry in list_of_entries:
            if not roots or roots[-1] != entry[0]:
                if debug==True:
                    print(entry[0])
                roots.append(entry[0])
        return roots
    
    if root_only==True: 
        return roots(list_of_entries, debug)
        
    return list_of_entries

## hard mode testing:
#process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")



def preprocess(text):

    remove_char_text = ''.join(c for c in text if c.isalpha() or c == "'" or c == ' ')

    print("processing", text)
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(remove_char_text)   

    splitted_text = remove_stopwords_list(splitted_text)    
    
    inflections = inflect(splitted_text) 
    
    entry_list = []
    entry_list = [entry[0] for entry in inflections if len(entry[0]) > 1]
    entry_list = [key for key, group in groupby(entry_list)]

    i = 0
    while i < len(entry_list) - 1:
        # Normalize words to form without diacritics
        word1 = unicodedata.normalize('NFD', entry_list[i])
        word2 = unicodedata.normalize('NFD', entry_list[i+1])

        # Remove diacritics
        word1 = ''.join(c for c in word1 if not unicodedata.combining(c))
        word2 = ''.join(c for c in word2 if not unicodedata.combining(c))

        if word1 in word2 or word2 in word1:
            if len(entry_list[i]) > len(entry_list[i+1]):
                entry_list.pop(i)
            else:
                entry_list.pop(i+1)
        else:
            i += 1
    print("processed", entry_list)
    return clean_results(entry_list)



#dict_names = ["AE", "AP90", "MW", "MWE"]
#path = "/resources/Sanskrit_dictionaries/"

def multidict(name: str, *args: str, source: str = "MW") -> Dict[str, Dict[str, List[str]]]:
    dict_names: List[str] = []
    dict_results: Dict[str, Dict[str, List[str]]] = {}
    name_component: str = ""
    
    # Collect dictionary names
    if not args:
        dict_names.append(source)
    else:
        dict_names.extend(args)
    
    session = Session()
    
    # For each dictionary, perform queries and process results
    for dict_name in dict_names:
        
        
        # Initial query
        query_builder = f"""
        SELECT keys_iast, components, cleaned_body FROM {dict_name + "clean"} 
        WHERE keys_iast = :name 
        OR keys_iast LIKE :wildcard_name
        """
        wildcard_name = f"{name}"
        
        with engine.connect() as connection:
            results = connection.execute(
                text(query_builder), 
                {"name": name, "wildcard_name": wildcard_name}
            ).fetchall()

        # Additional query if no results
        if not results and len(name) > 1:
            query_builder = f"""
            SELECT keys_iast, components, cleaned_body FROM {dict_name + "clean"} 
            WHERE keys_iast = :name 
            OR keys_iast LIKE :wildcard_name
            """
            wildcard_name = f"{name[:-1]}_"
            
            with engine.connect() as connection:
                results = connection.execute(
                    text(query_builder), 
                    {"name": name, "wildcard_name": wildcard_name}
                ).fetchall()
        
        #print(f"Results for {dict_name}: {results}")
        
        # Additional query if no results
        if not results and len(name) > 1:
            query_builder = f"""
            SELECT keys_iast, components, cleaned_body FROM {dict_name + "clean"} 
            WHERE keys_iast LIKE :name1 
            OR keys_iast LIKE :name2
            """
            with engine.connect() as connection:
                results = connection.execute(
                    text(query_builder), 
                    {"name1": name + "_", "name2": name[:-1] + "_"}
                ).fetchall()

        #print(f"Results for {dict_name} after second query: {results}")
        

        # Group results by components
        component_dict: Dict[str, List[str]] = {}
        for row in results:
            #print(f"Row: {row}")
            key_iast, components, cleaned_body = row
            if not name_component:
                name_component = components
            #print(f"key_iast: {key_iast}, components: {components}, cleaned_body: {cleaned_body}")
            if key_iast in component_dict:
                component_dict[key_iast].append(cleaned_body)
            else:
                component_dict[key_iast] = [cleaned_body]        
        # Add to dict_results
        dict_results[dict_name] = component_dict
    
    connection.close()
    return [name_component, dict_results]



# Example usage
#results = multidict("yoga", "MW" "AP90")
#print(results)





def get_mwword(word:str)->list[str, str, list[str]] : 
        session = Session()
        query_builder = text("SELECT components, cleaned_body FROM mwclean WHERE keys_iast = :word")
        print("query_builder", query_builder, word)
        results = session.execute(query_builder, {'word': word}).fetchall()
        session.close()        
        components = results[0][0]
        result_list = [row[1] for row in results]
        
        return [components, result_list]



def get_voc_entry(list_of_entries, *dict_names):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):

            word = entry[0]

            if '*' not in word and '_' not in word and '%' not in word:
                if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                    print("word", word)
                    entry = entry + multidict(word, *dict_names)
                else:
                    entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
                entries.append(entry)
            else:
                print("wildcard search", word)
                entry = entry + multidict(word, *dict_names)
                entries.append(entry)
            
        elif isinstance(entry, str):
            
            if '*' not in entry and '_' not in entry and '%' not in entry:
                if entry in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                    print("word", entry)
                    entry = [entry] + multidict(entry, *dict_names)
                else:
                    entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
                entries.append(entry)
            else:
                print("wildcard search", entry)
                entry = [entry] + multidict(entry, *dict_names)
                entries.append(entry)
    return entries



# find_inflection = False, inflection_table = False, 
#split_compounds = True, dictionary_search = False,
##first attempt to process the word not using the sandhi_splitter, which often gives uncorrect;
##then if the word is not found, try to split the word in its components and find the root of each component


import time

def processTimed(text, *dict_names, max_length=100, debug=True, root_only=False, recordTime=True):
    # Start timing overall execution
    start_process = time.time() if recordTime else None
    times = {} if recordTime else None

    ## if the text is too long, we try to trim it to the last whitespace
    start_time = time.time() if recordTime else None
    if len(text) > max_length:
        last_space_index = text[:max_length].rfind(' ')
        if last_space_index == -1:
            text = text[:max_length]
        else:
            # Trim up to the last whitespace
            text = text[:last_space_index]

    ## jj occours only in sandhi and we know it should be split into two words. 
    if 'jj' in text:
        text = text.replace('jj', 'j j')
    if recordTime:
        times['text_preprocessing'] = time.time() - start_time

    ## if the text is a single word, try to find the word in the dictionary for exact match, the split if it fails
    if ' ' not in text:
        start_time = time.time() if recordTime else None
        transliterated_text = anythingToSLP1(text)     
        text = regex.sub('[^\p{L}\']', '', transliterated_text)
        if recordTime:
            times['transliteration'] = time.time() - start_time

        start_time = time.time() if recordTime else None
        result = root_any_word(text)
        if recordTime:
            times['root_analysis'] = time.time() - start_time

        start_time = time.time() if recordTime else None
        if result is not None:
            for i, res in enumerate(result):
                if isinstance(res, str):
                    result[i] = transliterateSLP1IAST(res.replace('-', ''))
                elif isinstance(res, list):
                    if isinstance(res[0], str):
                        res[0] = transliterateSLP1IAST(res[0].replace('-', ''))
            result_vocabulary = get_voc_entry(result, *dict_names)
            if recordTime:
                times['dictionary_lookup'] = time.time() - start_time

            return clean_results(result_vocabulary, debug=debug, root_only=root_only)
        else:
            query = [transliterateSLP1IAST(text)]
            result_vocabulary = get_voc_entry(query, *dict_names)  
            if isinstance(result_vocabulary[0][2], dict):
                if recordTime:
                    times['total_time'] = time.time() - start_process
                return clean_results(result_vocabulary, debug=debug, root_only=root_only)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    start_time = time.time() if recordTime else None
    text = transliterateSLP1IAST(text)
    splitted_text = hybrid_sandhi_splitter(text)
    if recordTime:
        times['sandhi_splitting'] = time.time() - start_time

    start_time = time.time() if recordTime else None
    splitted_text = [transliterateIASTSLP1(word) for word in splitted_text]    
    inflections = inflect(splitted_text) 
    if recordTime:
        times['inflection_analysis'] = time.time() - start_time

    start_time = time.time() if recordTime else None
    inflections_vocabulary = get_voc_entry(inflections, *dict_names)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]       
    if recordTime:
        times['vocabulary_lookup'] = time.time() - start_time
        times['total_time'] = time.time() - start_process

    # Add timing data to the result if requested
    result = clean_results(inflections_vocabulary, debug=debug, root_only=root_only)
    if recordTime:
        if debug:
            print("\n=== Processing Time Summary ===")
            print(f"Total execution time: {times['total_time']:.4f} seconds")
            print("\nBreakdown by operation:")
            for operation, duration in times.items():
                if operation != 'total_time':
                    percentage = (duration / times['total_time']) * 100
                    print(f"{operation.replace('_', ' ').title()}: {duration:.4f}s ({percentage:.1f}%)")
        return result, times

    return result