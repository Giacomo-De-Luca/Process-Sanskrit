import pandas as pd
from process_sanskrit.utils.loadResources import get_resource_path

# Load stopwords using the resource path helper
stopwords = pd.read_csv(get_resource_path('stopwords.csv'))
stopwords_as_list = stopwords['stopword'].tolist()

def remove_stopwords(text_or_list):
    """
    Remove stopwords from either a string or a list of words.
    
    Args:
        text_or_list: Either a string or a list of words
        
    Returns:
        Either a string or a list, depending on the input type,
        with stopwords removed
    """
    if isinstance(text_or_list, list):
        return remove_stopwords_list(text_or_list)
    elif isinstance(text_or_list, str):
        return remove_stopwords_string(text_or_list)
    else:
        raise TypeError("Input must be either a string or a list")

def remove_stopwords_list(text_list):
    """Remove stopwords from a list of words"""
    return [word for word in text_list if word not in stopwords_as_list]

def remove_stopwords_string(text):
    """Remove stopwords from a string"""
    text = text.replace('.', '')  # Remove periods
    text_list = text.split()  # Split the string into words
    return ' '.join(word for word in text_list if word not in stopwords_as_list)
