import indic_transliteration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from .detectTransliteration import detect


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

def transliterate(text, transliteration_scheme):
    detected_scheme_str = detect(text).upper()
    transliteration_scheme_str = transliteration_scheme.upper()
    detected_scheme = getattr(sanscript, detected_scheme_str)
    output_scheme = getattr(sanscript, transliteration_scheme_str)
    return sanscript.transliterate(text, detected_scheme, output_scheme)