# -*- coding: utf-8 -*-
"""
    detect
    ~~~~~~

    Code for automatically detecting a transliteration scheme.

    :license: MIT and BSD
"""
import enum
import re
import json

#: Scheme data. This is split into separate classes, but here it's DRY.
TRANSLITERATION_SCHEMES = [
    ('Bengali', 0x0980),
    ('Devanagari', 0x0900),
    ('Gujarati', 0x0a80),
    ('Gurmukhi', 0x0a00),
    ('Kannada', 0x0c80),
    ('Malayalam', 0x0d00),
    ('Oriya', 0x0b00),
    ('Tamil', 0x0b80),
    ('Telugu', 0x0c00),
    ('HK', None),
    ('IAST', None),
    ('ITRANS', None),
    ('Kolkata', None),
    ('SLP1', None),
    ('Velthuis', None),
]

#: Start of the Devanagari block.
BRAHMIC_FIRST_CODE_POINT = 0x0900

#: End of the Malayalam block.
BRAHMIC_LAST_CODE_POINT = 0x0d7f

#: Schemes sorted by Unicode code point. Ignore schemes with none defined.
BLOCKS = sorted([x for x in TRANSLITERATION_SCHEMES if x[-1]], key=lambda x: -x[1])

class TransliterationScheme(enum.Enum): 
    Bengali = 'Bengali' 
    Devanagari = 'Devanagari' 
    Gujarati = 'Gujarati' 
    Gurmukhi = 'Gurmukhi' 
    Kannada = 'Kannada' 
    Malayalam = 'Malayalam' 
    Oriya = 'Oriya' 
    Tamil = 'Tamil' 
    Telugu = 'Telugu' 
    HK = 'HK' 
    IAST = 'IAST' 
    ITRANS = 'ITRANS' 
    Kolkata = 'Kolkata' 
    SLP1 = 'SLP1' 
    Velthuis = 'Velthuis'

class Regex:

    #: Match on special Roman characters
    IAST_OR_KOLKATA_ONLY = re.compile(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻ]')

    #: Match on chars shared by ITRANS and Velthuis
    ITRANS_OR_VELTHUIS_ONLY = re.compile(r'aa|ii|uu|~n')

    #: Match on ITRANS-only
    ITRANS_ONLY = re.compile(r'ee|oo|\^[iI]|RR[iI]|L[iI]|' \
                             r'~N|N\^|Ch|chh|JN|sh|Sh|\.a')

    #: Match on Kolkata-specific Roman characters
    KOLKATA_ONLY = re.compile(r'[ēō]')

    #: Match on SLP1-only characters and bigrams
    SLP1_ONLY = re.compile(r'[fFxXEOCYwWqQPB]|kz|Nk|Ng|tT|dD|Sc|Sn|' \
                           r'[aAiIuUfFxXeEoO]R|' \
                           r'G[yr]|(\W|^)G')

    #: Match on Velthuis-only characters
    VELTHUIS_ONLY = re.compile(r'\.[mhnrlntds]|"n|~s')


def detect(text):
    """Detect the input's transliteration scheme.

    :param text: some text data, either a `unicode` or a `str` encoded
                 in UTF-8.
    """
    # Verify encoding
   
    # Brahmic schemes are all within a specific range of code points.
    for L in text:
        code = ord(L)
        if code >= BRAHMIC_FIRST_CODE_POINT:
            for name, start_code in BLOCKS:
                if start_code <= code <= BRAHMIC_LAST_CODE_POINT:
                    return name

    # Romanizations
    if Regex.IAST_OR_KOLKATA_ONLY.search(text):
        if Regex.KOLKATA_ONLY.search(text):
            return TransliterationScheme.Kolkata.value
        else:
            #print(Scheme.__members__) 
            return TransliterationScheme.IAST.value

    if Regex.ITRANS_ONLY.search(text):
        return TransliterationScheme.ITRANS.value

    if Regex.SLP1_ONLY.search(text):
        return TransliterationScheme.SLP1.value

    if Regex.VELTHUIS_ONLY.search(text):
        return TransliterationScheme.Velthuis.value

    if Regex.ITRANS_OR_VELTHUIS_ONLY.search(text):
        return TransliterationScheme.ITRANS.value

    return TransliterationScheme.HK.value
