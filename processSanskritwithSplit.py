import indic_transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import ast
from detectTransliteration import detect
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
from typing import List, Dict, Tuple, Union

from dataclasses import dataclass

__version__ = "0.1"
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
    
def transliterateSLP1IAST(text):
    return transliterate(text, sanscript.SLP1, sanscript.IAST)   
def transliterateIASTSLP1(text):
    return transliterate(text, sanscript.IAST, sanscript.SLP1)   


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


def sandhi_splitter(text_to_split, cached=True, attempts=10):
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



from typing import List, Dict, Tuple
import ast
from dataclasses import dataclass
import re
from typing import List, Union, Tuple, Optional
import ast
from dataclasses import dataclass
from typing import List, Union, Tuple, Optional
import ast
from dataclasses import dataclass
@dataclass
class SplitResult:
    """Class to hold detailed results of a sandhi split with scoring information"""
    split: List[str]
    score: float
    subscores: dict
    all_splits: Optional[List[Tuple[List[str], float, dict]]] = None
from typing import List, Optional, Tuple, Dict
import ast

def enhanced_sandhi_splitter(
    text_to_split: str, 
    cached: bool = True, 
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
            
            scorer = SandhiSplitScorer()
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
                scorer = SandhiSplitScorer()
                score, subscores = scorer.score_split(simple_split)
                return simple_split, score, subscores, None
            return simple_split

        # Process splits based on attempts
        if attempts == 1:
            splits = [ast.literal_eval(f'{next(splits)}')]
        else:
            splits = [ast.literal_eval(f'{split}') for split in splits]

        # Score all splits
        scorer = SandhiSplitScorer()
        ranked_splits = scorer.rank_splits(splits)
        best_split, best_score, subscores = ranked_splits[0]
        
        # Cache the result if needed
        if cached:
            new_cache_entry = SplitCache(
                input=text_to_split, 
                splitted_text=str([split for split, _, _ in ranked_splits])
            )
            session.add(new_cache_entry)
            session.commit()
            print(f"Added to cache: {best_split}")

        if detailed_output:
            return best_split, best_score, subscores, ranked_splits
        return best_split

    except Exception as e:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {e}")
        simple_split = text_to_split.split()
        
        if detailed_output:
            scorer = SandhiSplitScorer()
            score, subscores = scorer.score_split(simple_split)
            return simple_split, score, subscores, None
        return simple_split


from typing import List, Dict, Tuple
import re

class SandhiSplitScorer:
    def __init__(self):
        # Common Sanskrit prefixes with weights
        self.common_prefixes = {
            'ā': 0.1,  # Simple prefixes get lower weights
            'ati': 0.2,
            'adhi': 0.2,
            'anu': 0.1,
            'apa': 0.2,
            'api': 0.1,
            'abhi': 0.2,
            'ava': 0.2,
            'ud': 0.1,
            'upa': 0.2,
            'dur': 0.2,
            'ni': 0.1,
            'nir': 0.2,
            'nis': 0.2,
            'parā': 0.2,
            'pari': 0.2,
            'pra': 0.2,
            'prati': 0.2,
            'vi': 0.1,
            'sam': 0.2,
            'su': 0.1
        }
        
        # Common Sanskrit suffixes with weights
        self.common_suffixes = {
            'sya': 0.1,
            'asya': 0.15,
            'āya': 0.15,
            'ena': 0.15,
            'eṣu': 0.15,
            'ebhyaḥ': 0.2,
            'āt': 0.1,
            'in': 0.1,
            'tva': 0.15,
            'tā': 0.15,
            'ka': 0.1
        }
        
        # Common Sanskrit compound patterns
        self.compound_patterns = [
            r'ānuśravika',  # Taddhita derivatives
            r'vitṛṣṇa',     # Negative compounds
            r'viṣaya'       # Common base words
        ]

    def score_split(self, split: List[str]) -> Tuple[float, Dict[str, float]]:
        scores = {}
        
        # 1. Compound recognition (new) - heavily weight recognition of valid compounds
        compound_score = 0
        for word in split:
            for pattern in self.compound_patterns:
                if re.search(pattern, word):
                    compound_score += 0.4  # High weight for recognized compounds
        scores['compounds'] = compound_score
        
        # 2. Length-based scoring - modified to prefer fewer splits more strongly
        length_score = 1.0 / (len(split) ** 1.5)  # Exponential penalty for more splits
        scores['length'] = length_score * 0.3
        
        # 3. Prefix/suffix recognition - with weighted importance
        prefix_suffix_score = 0
        if split[0] in self.common_prefixes:
            prefix_suffix_score += self.common_prefixes[split[0]]
        if split[-1] in self.common_suffixes:
            prefix_suffix_score += self.common_suffixes[split[-1]]
        scores['affixes'] = prefix_suffix_score * 0.2
        
        # 4. Word length distribution - modified to prefer more natural word lengths
        lengths = [len(word) for word in split]
        short_word_penalty = sum(1 for l in lengths if l < 2) * 0.1
        scores['word_lengths'] = max(0, 0.2 - short_word_penalty)
        
        # 5. Sanskrit morphology check (new)
        morphology_score = 0
        for word in split:
            # Reward typical Sanskrit word patterns
            if re.search(r'[āīūṛṝḷḹ]', word):  # Contains long vowels
                morphology_score += 0.1
            if len(word) >= 4:  # Reward reasonable word lengths
                morphology_score += 0.1
        scores['morphology'] = morphology_score * 0.2
        
        total_score = sum(scores.values())
        return total_score, scores

    def rank_splits(self, splits: List[List[str]]) -> List[Tuple[List[str], float, Dict[str, float]]]:
        scored_splits = []
        for split in splits:
            score, subscores = self.score_split(split)
            scored_splits.append((split, score, subscores))
        
        return sorted(scored_splits, key=lambda x: x[1], reverse=True)

    def rank_splits(self, splits: List[List[str]]) -> List[Tuple[List[str], float, Dict[str, float]]]:
        scored_splits = []
        for split in splits:
            score, subscores = self.score_split(split)
            scored_splits.append((split, score, subscores))
        
        return sorted(scored_splits, key=lambda x: x[1], reverse=True)
    

class SandhiSplitScorer:
    def __init__(self):
        # Common upasarga (verbal prefixes)
        self.upasargas = {
            'ā': 0.1, 'ati': 0.2, 'adhi': 0.2, 'anu': 0.1,
            'apa': 0.2, 'api': 0.1, 'abhi': 0.2, 'ava': 0.2,
            'ud': 0.1, 'upa': 0.2, 'dur': 0.2, 'ni': 0.1,
            'nir': 0.2, 'nis': 0.2, 'parā': 0.2, 'pari': 0.2,
            'pra': 0.2, 'prati': 0.2, 'vi': 0.1, 'sam': 0.2,
            'su': 0.1
        }
        
        # Common vibhakti (case endings)
        self.vibhaktis = {
            'sya': 0.1, 'asya': 0.15, 'āya': 0.15, 'ena': 0.15,
            'eṣu': 0.15, 'ebhyaḥ': 0.2, 'āt': 0.1
        }
        
        # Common taddhita (secondary derivative) suffixes
        self.taddhita_patterns = [
            r'[aā]ka$',    # -aka, -āka endings
            r'in$',        # -in endings
            r'[aā]na$',    # -ana, -āna endings
            r'tva$',       # -tva endings
            r'tā$',        # -tā endings
            r'maya$'       # -maya endings
        ]

    def score_split(self, split: List[str]) -> Tuple[float, Dict[str, float]]:
        scores = {}
        
        # 1. Length scoring - exponential decay for more splits
        length_score = 2 ** (-len(split) + 3)
        scores['length'] = length_score * 0.4
        
        # 2. Morphological analysis
        morphology_score = 0
        for word in split:
            # Check for valid taddhita formations
            for pattern in self.taddhita_patterns:
                if re.search(pattern, word):
                    morphology_score += 0.15
            
            # Reward words of typical Sanskrit length (not too short)
            if len(word) >= 3:
                morphology_score += 0.05
            
            # Check for standard Sanskrit syllable structure
            syllable_pattern = r'[aāiīuūṛṝḷḹeoṃḥ]'
            syllable_count = len(re.findall(syllable_pattern, word))
            if syllable_count >= 2:  # Reward multi-syllabic words
                morphology_score += 0.05
        
        scores['morphology'] = morphology_score * 0.3
        
        # 3. Sandhi analysis
        sandhi_score = 0
        joined = ' '.join(split)
        
        # Check for valid sandhi patterns at word boundaries
        for i in range(len(split) - 1):
            word1, word2 = split[i], split[i + 1]
            # Penalize unlikely vowel sequences across word boundaries
            if (word1[-1] in 'aāiīuūṛṝḷḹeoṃḥ' and 
                word2[0] in 'aāiīuūṛṝḷḹeoṃḥ'):
                sandhi_score -= 0.1
        
        scores['sandhi'] = max(0, (1 + sandhi_score)) * 0.3
        
        total_score = sum(scores.values())
        return total_score, scores

    def rank_splits(self, splits: List[List[str]]) -> List[Tuple[List[str], float, Dict[str, float]]]:
        scored_splits = []
        for split in splits:
            score, subscores = self.score_split(split)
            scored_splits.append((split, score, subscores))
        
        return sorted(scored_splits, key=lambda x: x[1], reverse=True)

def format_scored_split(split_info: Tuple[List[str], float, Dict[str, float]]) -> str:
    split, score, subscores = split_info
    result = f"Score: {score:.3f} | Split: {' + '.join(split)}\n"
    result += "Subscores:\n"
    for category, subscore in subscores.items():
        result += f"  {category}: {subscore:.3f}\n"
    return result


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
                query_builder2 = "SELECT * FROM lgtab1 WHERE stem = :root_form and model = :type "
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
            query_builder = "SELECT * FROM vlgtab2 WHERE key = :verb"
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
                query_builder2 = "SELECT * FROM vlgtab1 WHERE stem = :root_form and model = :type"
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

def root_any_word(word):


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


def dict_word_iterativeold(word, last_letter=None):
    sanskrit_end_letters = ["a", "A", "i", "I", "u", "U", "f", "d", "n", "t", "m", "s", "h"]  # Add all Sanskrit end-of-word letters here
    temp_word = word
    removed_part = ''  # Part of the word that was removed
    found = False  # Flag to indicate when a match has been found
    if temp_word in mwdictionaryKeysSLP1:  
            found_word = temp_word 
            found = True
            return [found_word, removed_part]
    while temp_word and not found: # Continue the loop until a match is found or temp_word is empty        
        if temp_word[:-1] in mwdictionaryKeysSLP1:  
            found_word = temp_word[:-1] 
            #print(temp_word[:-1] ) debug
            found = True
            return [found_word, removed_part]       
        elif temp_word[-1].isalpha():  # If the last character is a letter
            for letter in sanskrit_end_letters:
                attempt = temp_word[:-1] + letter
                #print(attempt) debug
                if attempt in mwdictionaryKeysSLP1:
                    found_word = attempt
                    found = True  # Set the flag to True when a match is found
                    break                                
        removed_part = temp_word[-1] + removed_part  # Keep track of the removed part
        temp_word = temp_word[:-1]  # Remove the last character, regardless of whether it's a letter or not
    
    return [found_word, removed_part] if found else None  # Return the match if found, else return None





def dict_word_iterative3(word, last_letter=None):
    """
    Parse Sanskrit compound words with inflection handling and multi-word compounds.
    Handles both SLP1 and IAST transliteration.
    
    Args:
        word (str): The word to parse
        last_letter (str): The last letter of the previous word (optional)
    
    Returns:
        list: List of [found_word, removed_part, inflection] triples for each identified component
    """
    # Common inflectional endings
    inflectional_endings = {
        # Nominal endings (organized by length to ensure longer matches first)
        'dual_instr_dat_abl': ['bhyAm', 'bhyām'],
        'plural_instr': ['bhiH', 'bhiḥ'],
        'plural_dat_abl': ['bhyaH', 'bhyaḥ'],
        'plural_gen': ['nAm', 'nām'],
        'singular_instr': ['ena', 'ayA', 'ayā'],
        'dual_nom_acc': ['au'],
        'singular_loc': ['i', 'e'],
        'singular_gen': ['sya', 'yAH', 'yāḥ'],
        'singular_dat': ['Aya', 'āya', 'yai'],
        'plural_loc': ['su', 'ṣu'],
        'singular_acc': ['am', 'ām'],
    }
    
    def remove_inflection(word):
        """
        Attempt to remove inflectional endings from a word.
        Returns: (stem, ending) or (word, '') if no ending found
        """
        for category, endings in inflectional_endings.items():
            for ending in endings:
                if word.endswith(ending):
                    return word[:-len(ending)], ending
        return word, ''
    
    def find_component(word_part):
        """
        Find a valid dictionary component, trying both with and without inflection.
        Returns: (found_word, remaining, inflection) or None
        """
        if not word_part:
            return None
            
        # First try the whole word
        if word_part in mwdictionaryKeysSLP1:
            return word_part, '', ''
            
        # Then try removing inflection
        stem, inflection = remove_inflection(word_part)
        if stem in mwdictionaryKeysSLP1:
            return stem, '', inflection
            
        # Try progressive character removal
        temp_word = word_part
        removed = ''
        while len(temp_word) > 1:  # Keep at least one character
            if temp_word in mwdictionaryKeysSLP1:
                return temp_word, removed, ''
            removed = temp_word[-1] + removed
            temp_word = temp_word[:-1]
            
            # Check with inflection removal at each step
            stem, infl = remove_inflection(temp_word)
            if stem in mwdictionaryKeysSLP1:
                return stem, removed, infl
                
        return None
    
    # Main processing
    results = []
    remaining_word = word
    last_unmatched = ''
    
    while remaining_word:
        result = find_component(remaining_word)
        if result:
            found_word, removed_part, inflection = result
            if last_unmatched:  # Handle any unmatched characters from previous iterations
                removed_part = last_unmatched + removed_part
                last_unmatched = ''
            results.append([found_word, removed_part, inflection])
            remaining_word = removed_part
        else:
            # If no valid component found, remove one character and continue
            if len(remaining_word) > 1:
                last_unmatched = remaining_word[-1] + last_unmatched
                remaining_word = remaining_word[:-1]
            else:
                # Handle the last character if no match found
                last_unmatched = remaining_word + last_unmatched
                if results:
                    results[-1][1] = last_unmatched + results[-1][1]
                else:
                    results.append(['', last_unmatched, ''])
                break
    
    return results if results else None

        ##if the dictionary approach fails, try the iterative approach:


            

def root_compoundsold(word):
    if word.startswith("'"):
        word = 'a' + word[1:]

    # Base case: if word is empty, return empty list
    if not word:
        return []

    roots = []

    first_root = dict_word_iterativeold(word)
    if not first_root or not first_root[0]:
        # No root found
        return []
    print("first_root", first_root)
    first_root_entry = root_any_word(first_root[0])
    print("first_root_entry", first_root_entry)
    if first_root_entry is not None:
        if isinstance(first_root_entry, list):
            roots.extend(first_root_entry)
        else:
            roots.append(first_root_entry)
    else:
        roots.append(first_root[0])

    # Remove the first root from the word
    without_root = word.replace(first_root[0], '', 1)

    # If there's nothing left, return the roots found
    if not without_root:
        return roots

    # Check if the word is being reduced
    if len(without_root) >= len(word):
        # Word not reduced; cannot proceed further
        return roots

    # Now, recursively process the remaining word
    rest_entries = root_any_word(without_root)
    if rest_entries is not None:
        if isinstance(rest_entries, list):
            roots.extend(rest_entries)
        else:
            roots.append(rest_entries)
    else:
        # Try dict_word_iterative and ensure result is longer than 2
        rest_root = dict_word_iterative(without_root)
        if rest_root is None or len(rest_root[0]) < 2:
            # Cannot proceed further; return roots found so far
            return roots
        else:
            # Check again if the word is being reduced
            if len(without_root) >= len(word):
                return roots
            # Recursively process the rest of the word
            rest_roots = root_compounds(without_root)
            roots.extend(rest_roots)

    return roots


def dict_word_iterative(word, last_letter=None):
    """
    Enhanced iterative Sanskrit word parser that combines dictionary lookup with 
    inflection handling and letter substitution.
    
    Args:
        word (str): The word to parse
        last_letter (str): The last letter of the previous word for sandhi analysis
    
    Returns:
        list: [found_word, removed_part] if match found, None otherwise
    """
    # Core Sanskrit endings and their variations
    sanskrit_end_letters = ["a", "A", "i", "I", "u", "U", "f", "d", "n", "t", "m", "s", "h"]
    
    # Common inflectional endings ordered by length (longest first)
    inflectional_endings = [
        'bhyAm', 'bhiH', 'bhyaH', 'nAm',  # Longer endings
        'ena', 'ayA', 'Aya', 'sya', 'yAH', 
        'au', 'am', 'An', 'su', 'i', 'e'   # Shorter endings
    ]
    
    def try_remove_inflection(word):
        """Helper function to attempt inflection removal."""
        for ending in inflectional_endings:
            if word.endswith(ending):
                return word[:-len(ending)]
        return word
    
    def check_dictionary(word):
        """Helper function to check word against dictionary with letter variations."""
        if word in mwdictionaryKeysSLP1:
            return word
            
        # Try with inflection removed
        stem = try_remove_inflection(word)
        if stem in mwdictionaryKeysSLP1:
            return stem
            
        # Try letter substitutions
        if word[-1].isalpha():
            for letter in sanskrit_end_letters:
                attempt = word[:-1] + letter
                if attempt in mwdictionaryKeysSLP1:
                    return attempt
                    
                # Try also with inflection removed after substitution
                stem = try_remove_inflection(attempt)
                if stem in mwdictionaryKeysSLP1:
                    return stem
        return None

    # Main processing
    temp_word = word
    removed_part = ''
    found = False
    found_word = None
    
    # First try the complete word
    match = check_dictionary(temp_word)
    if match:
        return [match, removed_part]
        
    # Iterative reduction and checking
    while temp_word and not found:
        # Try current word
        match = check_dictionary(temp_word)
        if match:
            found_word = match
            found = True
            break
            
        # Remove last character and continue
        removed_part = temp_word[-1] + removed_part
        temp_word = temp_word[:-1]
    
    return [found_word, removed_part] if found else None

def root_compounds2(word):
    """
    Enhanced compound word analyzer that recursively breaks down Sanskrit compounds
    into their constituent parts.
    
    Args:
        word (str): The compound word to analyze
        
    Returns:
        list: List of identified roots/stems
    """
    # Handle initial apostrophe (avagraha)
    if word.startswith("'"):
        word = 'a' + word[1:]

    # Base case
    if not word:
        return []
        
    roots = []
    
    # Try to find the first component
    first_component = dict_word_iterative(word)
    if not first_component or not first_component[0]:
        return []
        
    # Process the first component
    first_root_entry = root_any_word(first_component[0])
    if first_root_entry is not None:
        if isinstance(first_root_entry, list):
            roots.extend(first_root_entry)
        else:
            roots.append(first_root_entry)
    else:
        roots.append(first_component[0])
    
    # Process remaining part
    remaining = word[len(first_component[0]):]
    if not remaining:
        return roots
        
    # Ensure we're making progress
    if len(remaining) >= len(word):
        return roots
        
    # Process the rest recursively
    rest_entries = root_any_word(remaining)
    if rest_entries is not None:
        if isinstance(rest_entries, list):
            roots.extend(rest_entries)
        else:
            roots.append(rest_entries)
    else:
        # Try iterative parsing if direct lookup fails
        rest_component = dict_word_iterative(remaining)
        if rest_component and len(rest_component[0]) >= 2:
            rest_roots = root_compounds(remaining)
            roots.extend(rest_roots)
    
    return roots



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
    'j': ['k', 'g', 'j'],
    'z': ['s', 'z', 'S']
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


def root_compounds(word, debug=False):
    """
    Analyzes a long Sanskrit compound with improved sandhi handling between segments.
    """
    if debug:
        print("\nStarting analysis of:", word)
        print("Length:", len(word))
        
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
        
        # Process the matched word
        root_entry = root_any_word(matched_word)
        if root_entry:
            if isinstance(root_entry, list):
                roots.extend(root_entry)
            else:
                roots.append(root_entry)
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



def inflect(splitted_text):
    roots = []
    prefixes = ['sva', 'anu', 'sam', 'pra', 'upa', 'vi', 'nis', 'abhi', 'ni', 'pari', 'prati', 'parA', 'ava', 'adhi', 'api', 'ati', 'ud', 'dvi', 'su', 'dur', 'duḥ']  # Add more prefixes as needed
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





def process(text, *dict_names, max_length=80):


    ## if the text is too long, we try to trim it to the last whitespace
    if len(text) > max_length:
        last_space_index = text[:max_length].rfind(' ')
        if last_space_index == -1:
            text = text[:max_length]
        else:
            # Trim up to the last whitespace
            text = text[:last_space_index]

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

        ## do some preliminary cleaning using sandhi rules
        if text[0] == "'":
            text = 'a' + text[1:]
        if text[-1] == 'o':
            text = text[:-1] + 'aH'
        elif text[-1] == 'S':
            text = text[:-1] + 'H'
        elif text[-1] == 'y':
            text = text[:-1] + 'i'
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
                    else:
                        for nested_prefix in sorted(SANSKRIT_PREFIXES.keys(), key=len, reverse=True):
                            if remainder.startswith(nested_prefix):
                                nested_remainder = remainder[len(prefix):]
                                nested_attempt = root_any_word(nested_remainder)
                                if nested_attempt is not None:
                                    result = [prefix] + [nested_prefix] + nested_attempt
                                    break

        ## if the word ends with a M, try to correct it for different conventions  and find the word in the dictionary for exact match
        if result is None and text[-1] == 'M':
            tentative = text[:-1] + 'm'
            attempt = root_any_word(tentative)
            if attempt is not None:
                result = attempt

        ## if the words starts with C, try to find out if it's the sandhied form of a word starting with S
        if result is None and text[0:1] == "C":
            print("tentative", text)
            tentative = 'S' + text[1:] 
            attempt = root_any_word(tentative)
            print("attempt", attempt)
            if attempt is not None:
                result = attempt

        if result is not None:
            print("Getting some results with no splitting here:", result)

            for res in result:
                if isinstance(res, list):
                    res[0] = transliterateSLP1IAST(res[0].replace('-', ''))
                    #print("res", res)
            result_vocabulary = get_voc_entry(result, *dict_names)  

            ## if the word is inside the dictionary, we return the entry directly, since it will be accurate. 
            if isinstance(result_vocabulary, list):
                if result[0][0] != result[0][4] and result[0][4] in mwdictionaryKeys:
                    replacement = get_voc_entry([result[0][4]], *dict_names)
                    print("replacement", replacement[0])
                    print("len replacement", len(replacement[0]))
                    if replacement is not None:
                        result_vocabulary.insert(0, replacement[0])

            #print("result_vocabulary", result_vocabulary)
            return clean_results(result_vocabulary)
        else:
            query = [transliterateSLP1IAST(text)]
            print("query", query)
            result_vocabulary = get_voc_entry(query, *dict_names)  
            #print("result_vocabulary", result_vocabulary)
            if isinstance(result_vocabulary[0][2], dict):
            #result_vocabulary[0][0] != result_vocabulary[0][2][0]:
                return clean_results(result_vocabulary)
    
    ## given that the text is composed of multiple words, we split them first then analyse one by one
    ## attempt to remove sandhi and tokenise in any case
    print("pre_splitted_text", text)
    text = transliterateSLP1IAST(text)
    #print("transliterate to split", text)
    splitted_text = enhanced_sandhi_splitter(text, cached=False)
    splitted_text = [transliterateIASTSLP1(word) for word in splitted_text]    
    inflections = inflect(splitted_text) 
    inflections_vocabulary = get_voc_entry(inflections, *dict_names)
    inflections_vocabulary = [entry for entry in inflections_vocabulary if len(entry[0]) > 1]       

    return clean_results(inflections_vocabulary)


##process("dveṣānuviddhaścetanācetanasādhanādhīnastāpānubhava")

filtered_words = ["ca", "na", "eva", "ni", "apya", "ava", "sva"]


def clean_results(list_of_entries):

    i = 0

    #print("it breaks right here:", list_of_entries)

    if len(list_of_entries[i]) == 7 and list_of_entries[i][0][-1] == "n" and list_of_entries[i][4] != list_of_entries[i][0]:
        if list_of_entries[i][4] in mwdictionaryKeys:
            replacement = get_voc_entry([list_of_entries[i][4]])
            if replacement is not None:
                list_of_entries[i] = replacement[0]

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
        
        # Check if the word is "sam"
        if list_of_entries[i][0] == "sam":
            j = i + 1
            while j < len(list_of_entries) and (list_of_entries[j][0] == "sa" or list_of_entries[j][0] == "sam"):
                j += 1
            if j < len(list_of_entries):
                voc_entry = get_voc_entry(["sam" + list_of_entries[j][0]])
                print("voc_entry", voc_entry)
                if voc_entry[0][0] == voc_entry[0][2][0]:
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
    for entry in list_of_entries:
        print(entry[0])

    return list_of_entries

def process_test(text, remove_stopwords = False, dictionary_entry = True, output_encoding = "IAST", entry_only = False):
    
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(text)
    
    ## removes stopwords
    if remove_stopwords == True: 
        splitted_text = remove_stopwords_list(splitted_text)
        
    
    inflections = inflect(splitted_text) 
    
    if dictionary_entry == True:
        inflections = get_voc_entry(inflections)    

    if entry_only == True:
        entry_list = []
        for entry in inflections:
            entry_list.append(entry[0])
        inflections = entry_list    

    return clean_results(inflections)

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



def get_voc_entry(list_of_entries, *args: str, source: str = "MW"):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):
            
            word = entry[0]
            dict_entry = []
            if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                key2 = dictionary_json[word][0][1] ##qui si dovrebbe fare un fetch SQL 
                key2 = transliterateSLP1IAST(key2) ## qui ho un leggero problema, se prendo l'equivalente in SQL da dove prendo la key 2? dalla prima o dalla seconda voce. Possibilmente dalla seconda se c'è più di una voce. 
    
                for entry_dict in dictionary_json[word]:
                    dict_entry.append(re.sub(r'<s>(.*?)</s>', lambda m: '<s>' + transliterateSLP1IAST(m.group(1)) + '</s>', entry_dict[4]))
                entry.append(key2)
                entry.append(dict_entry)
            else:
                entry.append(word)  # Append the original word for key2
                entry.append([word])  # Append the original word for dict_entry
            entries.append(entry)
            
        elif isinstance(entry, str):
            
            dict_entry = []
            if entry in mwdictionaryKeys:  # Check if the key exists
                key2 = dictionary_json[entry][0][1]
                key2 = transliterateSLP1IAST(key2)
    
                for entry_dict in dictionary_json[entry]:
                    dict_entry.append(re.sub(r'<s>(.*?)</s>', lambda m: '<s>' + transliterateSLP1IAST(m.group(1)) + '</s>', entry_dict[4]))
                entry = [entry]    
                
                entry.append(key2)
                entry.append(dict_entry)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
    return entries

# find_inflection = False, inflection_table = False, 
#split_compounds = True, dictionary_search = False,
##first attempt to process the word not using the sandhi_splitter, which often gives uncorrect;
##then if the word is not found, try to split the word in its components and find the root of each component


def get_mwword(word:str)->list[str, list[str]] : 
        session = Session()
        query_builder = "SELECT key2, cleaned_body FROM mwclean WHERE keys_iast = :word"
        results = session.execute(query_builder, {'word': word}).fetchall()
        session.close()        
        components = results[0][0]
        result_list = [row[1] for row in results]
        
        return [components, result_list]




def get_voc_entry(list_of_entries):
    entries = []
    for entry in list_of_entries:        
        if isinstance(entry, list):

            word = entry[0]

            if word in mwdictionaryKeys:  # Check if the key exists ## check non nel dizionario, ma solo nella lista chiavi
                entry = entry + get_mwword(word)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
            
        elif isinstance(entry, str):
            
            if entry in mwdictionaryKeys:  # Check if the key exists
                entry = [entry] + get_mwword(entry)
            else:
                entry = [entry, entry, [entry]]  # Append the original word for key2 and dict_entry
            entries.append(entry)
    return entries

# find_inflection = False, inflection_table = False, 
#split_compounds = True, dictionary_search = False,
##first attempt to process the word not using the sandhi_splitter, which often gives uncorrect;
##then if the word is not found, try to split the word in its components and find the root of each component



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

