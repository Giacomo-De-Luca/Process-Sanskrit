import indic_transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate as indic_transliterate
from .detectTransliteration import detect


##to get all the available schemes
##indic_transliteration.sanscript.SCHEMES.keys()


## Glyphs that stand in for the avagraha (the elided initial *a-*) in the wild.
## Editions, OCR passes and PDF copy-paste each pick a different one, and only
## the ASCII apostrophe is understood downstream.  Fold them all onto it before
## anything else looks at the text.
##
## The two failure modes this prevents are different, and both end in a
## shattered word:  U+2019/U+2018/U+0060/U+00B4 are punctuation, so preprocess's
## \p{L} filter deletes them outright;  U+02BC is category Lm -- a *letter* --
## so it survives that filter and reaches the splitter as a bogus consonant.
AVAGRAHA_VARIANTS = {
    "'": "'",   # U+0027 apostrophe (the canonical form, kept as-is)
    "’": "'",   # U+2019 right single quotation mark -- the usual PDF/OCR glyph
    "‘": "'",   # U+2018 left single quotation mark
    "ʼ": "'",   # U+02BC modifier letter apostrophe (a letter! see above)
    "ʻ": "'",   # U+02BB modifier letter turned comma
    "`": "'",   # U+0060 grave accent
    "´": "'",   # U+00B4 acute accent
    "′": "'",   # U+2032 prime
}

_AVAGRAHA_TABLE = str.maketrans(AVAGRAHA_VARIANTS)


def normalize_avagraha(text):
    """
    Fold every apostrophe-like avagraha glyph onto the ASCII apostrophe.

    Args:
        text: text in any transliteration scheme.

    Returns:
        The same text with all AVAGRAHA_VARIANTS replaced by "'".

    Example:
        >>> normalize_avagraha("so’nupalambhena")
        "so'nupalambhena"
    """
    return text.translate(_AVAGRAHA_TABLE)


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