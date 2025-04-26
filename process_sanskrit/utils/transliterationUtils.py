import indic_transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate as indic_transliterate
from .detectTransliteration import detect


##to get all the available schemes
##indic_transliteration.sanscript.SCHEMES.keys()

def transliterate(text, transliteration_scheme, input_scheme=None):
    """
    Transliterate text from one scheme to another.
    
    Args:
        text (str): The text to transliterate
        transliteration_scheme (str): Target scheme (e.g., "SLP1", "IAST", "HK", "DEVANAGARI")
        input_scheme (str, optional): Source scheme. If None, will auto-detect.
    
    Returns:
        str: Transliterated text
        
    Examples:
        # SLP1 to IAST
        transliterate("rAma", "IAST", "SLP1")  # "rāma"
        
        # Auto-detect to SLP1
        transliterate("रामः", "SLP1")  # "rAmaH"
        
        # Auto-detect to IAST
        transliterate("rAma", "IAST")  # "rāma"
        
        # SLP1 to HK
        transliterate("rAma", "HK", "SLP1")  # "raama"
        
        # DEVANAGARI to SLP1
        transliterate("राम", "SLP1", "DEVANAGARI")  # "rAma"
    """

    if not input_scheme:
        detected_scheme_str = detect(text).upper()
        transliteration_scheme_str = transliteration_scheme.upper()
        input_scheme = getattr(sanscript, detected_scheme_str)
        output_scheme = getattr(sanscript, transliteration_scheme_str)
    else: 
        input_scheme_str = input_scheme.upper()
        input_scheme = getattr(sanscript, input_scheme_str)
        transliteration_scheme_str = transliteration_scheme.upper()
        output_scheme = getattr(sanscript, transliteration_scheme_str)

    return indic_transliterate(text, input_scheme, output_scheme)