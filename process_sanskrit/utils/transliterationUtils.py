import regex

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


## An avagraha marks an *a-* elided after a preceding **e** or **o**, and after
## nothing else -- that is the whole of the sandhi rule.  So position, not the
## glyph, is what identifies one: an apostrophe anywhere else is a quotation
## mark or OCR noise, and turning it into a vowel silently corrupts the word
## (`iti ‘yoga’ ucyate` would otherwise yield the real-but-wrong lemma *ayoga*).
## A leading apostrophe is ambiguous -- "'nupalambhena" is an avagraha, "'tapas'"
## is a quotation -- and what tells them apart is that a quotation gets *closed*.
## Strip balanced quotes before the rules below read anything as an elision.
_BALANCED_QUOTES = regex.compile(r"^'(.+)'$")

_AVAGRAHA_AFTER_O = regex.compile(r"(\p{L}*o)\s*'")
_AVAGRAHA_AFTER_E = regex.compile(r"(\p{L}*e)\s*'")
_AVAGRAHA_INITIAL = regex.compile(r"^'")
_LEFTOVER_APOSTROPHE = regex.compile(r"'")

## A word-final -o before an avagraha normally comes from -aḥ/-as, and is undone
## along with the elision (saḥ + anupalambhena -> so 'nupalambhena).  These
## indeclinables are the exception: their -o is original and must be kept, so
## only the elided a- is restored (aho 'yam -> aho ayam, never *ahaḥ ayam).
O_NOT_FROM_VISARGA = frozenset({"o", "aho", "bho", "ho"})


def _restore_after_o(match):
    word = match.group(1)
    if word.lower() in O_NOT_FROM_VISARGA:
        return f"{word} a"
    return f"{word[:-1]}aḥ a"


def restore_avagraha(text):
    """
    Undo avagraha elision: put the elided initial *a-* back on the word.

    Expects IAST.  Handles every glyph in AVAGRAHA_VARIANTS, spaced or not,
    and undoes the -aḥ/-as -> -o sandhi on the preceding word where that is
    what produced the o.  Any apostrophe that is not in an avagraha position is
    dropped rather than passed on -- notably U+02BC, which is a Unicode *letter*
    and would otherwise reach the splitter as a bogus consonant.

    Args:
        text: IAST text, avagrahas written with any of AVAGRAHA_VARIANTS.

    Returns:
        The text with elided vowels restored and stray apostrophes removed.

    Examples:
        >>> restore_avagraha("so 'nupalambhena")
        'saḥ anupalambhena'
        >>> restore_avagraha("te’pi")
        'te api'
        >>> restore_avagraha("aho 'yam")
        'aho ayam'
        >>> restore_avagraha("iti ‘yoga’ ucyate")
        'iti yoga ucyate'
    """
    text = normalize_avagraha(text)
    text = _BALANCED_QUOTES.sub(r"\1", text)
    text = _AVAGRAHA_AFTER_O.sub(_restore_after_o, text)
    text = _AVAGRAHA_AFTER_E.sub(r"\1 a", text)
    text = _AVAGRAHA_INITIAL.sub("a", text)
    return _LEFTOVER_APOSTROPHE.sub("", text)


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