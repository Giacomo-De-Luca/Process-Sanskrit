### 
#   
#   here are a list of dictionaries that are used in the sandhi module
#   variableSandhiSLP1: dictionary of sandhi variations for word-final consonants
#   sanskritFixedSandhiMapSLP1: dictionary of fixed sandhi variations for word-final consonants
#   FINAL_AU_SPELLING, read_final_sandhi: a word's last letter read as the sound it stands for
#   VOWEL_SANDHI_INITIALS: dictionary of sandhi variations for word-final vowels
#   SANDHI_VARIATIONS: dictionary of sandhi variations for word-final letters
#   SANDHI_VARIATIONS_IAST: dictionary of sandhi variations for word-final letters in IAST notation
#   SANSKRIT_PREFIXES: dictionary of common Sanskrit prefixes
#   PARTICLE_PREFIXES: the prefixes above that running text uses as particles (api)
#   PREFIX_LOOKUP_SPELLINGS: the spelling under which a stripped prefix is looked up
#   SUPPLETIVE_ROOTS: the root a reader looks up for a verb the table files elsewhere
#   and the __all__ list of variables that are exported by this module
#
#  
###


variableSandhi = {
    'k': ['t', 'c', 'ś'],        # ṝrom t/c/ś in word-final position
    'ṭ': ['t', 'ś', 'ṣ'],        # ṝrom t/ś/ṣ in word-final position
    't': ['d'],                   # ṝrom d in word-final position
    'n': ['t', 'm', 'ṃ'],        # ṝrom t/m/aṃ in word-final position
    'p': ['t', 'b'],             # ṝrom t/b in word-final position
    'ḍ': ['t', 'd'],             # ṝrom t/d in word-final position
    's': ['ś', 'ṣ', 'ḥ'],        # ṝrom ś/ṣ/ḥ in word-final position
    'ś': ['k', 'ḥ'],             # ṝrom k/ḥ in word-final position
    'ṣ': ['k', 'ḥ', 't'],        # ṝrom k/ḥ/t in word-final position
    'r': ['ḥ', 's'],             # ṝrom ḥ/s in word-final position
    'o': ['aḥ', 'as', 'au'],     # ṝrom as/aḥ/au in word-final position
    'j': ['t'], 
    'ṃ': ['m'],
    'c': ['t'],
    ## vowel sandhi, to add in detail here
    'ā': ['āḥ']

}

sanskritFixedSandhiMap = {
    'y': 'i',          # 'ī' out ṛor testing # y comes ṛrom i or ī beṛore voṭels (like devī + atra → devyatra)
    'r': 'ḥ',          # r comes ṛrom visarga beṛore voiced sounds and some voṭels (like punaḥ + gacchati → punargacchati)
    #'ṅ': 'n',          # ṅ comes ṛrom n beṛore velars (k, kh, g, gh) (like tān + karoti → tāṅkaroti)
    #'ñ': 'n',          # ñ comes ṛrom n beṛore palatals (c, ch, j, jh) (like tān + carati → tāñcarati)
    #'ṇ': 'n',          # ṇ comes ṛrom n beṛore retroṛleḷes (ṭ, ṭh, ḍ, ḍh) (like tān + ṭīkate → tāṇṭīkate)
    'v': 'u',          # 'ū' out ṛor testing  # v comes ṛrom u or ū beṛore voṭels (like guru + atra → gurvatra)
    'd': 't',          # d comes ṛrom t beṛore voiced consonants (like tat + dānam → taddānam)
    'b': 'p',          # b comes ṛrom p beṛore voiced consonants (like ap + bhiḥ → abbhiḥ)
    'g': 'k',          # g comes ṛrom k beṛore voiced consonants (like vāk + devi → vāgdevi)
    'ś': 'ḥ',

}

## A final `-āv` is `-au` before a vowel (`samādhau eva` -> `samādhāv eva`).
## The map above reads a final `v` as `u`, which would make it `-āu`, a form no
## table has: no inflected form ends in `v` or in `-āu`.  See
## documentation/final-av.md.
FINAL_AU_SPELLING = ('āv', 'au')


def read_final_sandhi(word):
    """`word` with its last letter read as the sound it stands for in pausa.

    `samādhāv` -> `samādhau`, `gurv` -> `guru`, `ity` -> `iti`, `punar` ->
    `punaḥ`; a word whose last letter the map does not name is returned as is.
    """
    spelling, sound = FINAL_AU_SPELLING
    if word.endswith(spelling):
        return word[:-len(spelling)] + sound
    if word and word[-1] in sanskritFixedSandhiMap:
        return word[:-1] + sanskritFixedSandhiMap[word[-1]]
    return word







# Dictionary mapping final vowels to possible initial vowels in SLP1 notation
VOWEL_SANDHI_INITIALS = {
    # when a word ends in 'ā', the next word might have lost initial 'a' or 'ā'
    'ā': ['a', 'ā'],
    
    # for final 'a', check for lost initial 'i'/'ī' (e) or 'u'/'ū' (o)
    #'a': ['i', 'ī', 'u', 'ū'],
    
    # for final 'i'/'ī', the next word might have lost initial 'i'/'ī'
    #'i': ['i',],
    'ī': ['i', 'ī'],

   # 'i': ['i', 'e'],
    
    # for final 'u'/'ū', the next word might have lost initial 'u'/'ū'
    #'u': ['u', 'ū'],
    'ū': ['u', 'ū'],
    
    # for final 'e', check the lost initial 'a'/'ā'
    'e': ['i'],
    
    # for final 'o', check the lost initial 'a'/'ā'
    'o': ['u'],

}

# New dictionary for sandhi variations in final letters
SANDHI_VARIATIONS = {

    # Vowel variations
    'ā': ['a', 'ā'],
    'ī': ['i', 'ī'],
    'ū': ['u', 'ū'],
    'ch': ['ś'],

    # Visarga variations
    'ḥ': ['s', 'r', 'ḥ'],
    'o': ['a', 'ā', 'o', 'u'],
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

    'ch': ['t', 'ś'], # poor cases like tacchabdaḥ

'    ṅ': ['n', 'ṅ'], # poor cases like saṅgacchati

    'ṃ': ['m', 'n'],
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


## Do NOT add `nir` here.  root_any_word strips these before process() tries the
## whole-word dictionary lookup and the -tā/-tva deriver, so a strippable `nir`
## pre-empts both: nirvikāratvāt comes back as nis + vikāratva instead of
## nirvikāratva.  The prefix `nis` is read off the splitter's `niḥ` token
## instead -- see Inflector.SPLIT_TOKEN_SPELLINGS and
## documentation/prefix-segmentation.md.
SANSKRIT_PREFIXES = [
    'sam', 'saṃ', 'anu', 'abhi', 'ati', 'adhi', 'apa', 'api', 'ava', 'ā', 'a', 'ud', 'upa', 'nis', 'parā', 'pari', 'pra', 'prati', 'praty', 'vi', 'vy', 'ut', 'ni'
]

## Prefixes above that running text nearly always uses as particles.  `api` is
## stripped and cut off like a preverb, but in `nāpyasti` it is the particle
## "also, even", a word of its own.  Lemma mode does not tie these to the part
## after them with a joined key: the 8 parts of the 1,500 held-out tokens that it
## keyed with `api+` were all the particle in the gold segmentation.
PARTICLE_PREFIXES = {'api'}

## Some of the spellings above are not words.  root_any_word builds a stripped
## prefix's own entry by looking the prefix up in the forms table, and `vy` is
## only what `vi` looks like before a vowel: looked up as written it resolves to
## nothing, and the prefix used to vanish from the analysis without a trace
## (vyavaharati -> avahṛ, a real verb and the wrong one).  Each such spelling is
## mapped here to the one the forms table files the prefix under.
##
## Every prefix above must resolve to its own entry, as written or through this
## map -- tests/test_prefix_segmentation.py checks it -- so a sandhi spelling
## added to the list needs an entry here too.  Without one it is dropped, or
## read as whatever unrelated word happens to share the spelling.
PREFIX_LOOKUP_SPELLINGS = {
    'ud': 'ut',        # the table files this prefix under its pausa form
    'vy': 'vi',
    'praty': 'prati',
}

## Verbs whose forms the verb table files under a root other than the one a
## reader looks them up under.  `paśyati` is the present of `dṛś`, but the table
## files that present under the root `paś`, so it reached `paś` only.  Lemma
## mode adds the root mapped here right after the one the table gives, which
## stays (`paś`, `dṛś`); see documentation/lemma-mode.md.
##
## Add an entry only after checking it against the data.  The other suppletive
## verbs checked already come back under the root a reader looks for: `uvāca`
## -> `vac`, `bravīti` -> `brū`, `avadhīt` and `jaghāna` -> `han`.
SUPPLETIVE_ROOTS = {
    'paś': 'dṛś',
}

SANSKRIT_PREFIXES_OLD = {
    'sam': 'together, completely',
    'saṃ': 'together, completely',    
    'anu': 'along, aṛter',
    'abhi': 'toṭards, into',
    'ati': 'beyond, over',
    'adhi': 'over, upon',
    'apa': 'aṭay, oṛṛ',
    'api': 'unto, close',
    'ava': 'doṭn, oṛṛ',
    'ā': 'near to, completely',
    'a': 'near to, completely',
    'ud': 'up, upṭards',
    'upa': 'toṭards, near',
    'nis': 'out, aṭay',
    'parā': 'aṭay, back',
    'pari': 'around, about',
    'pra': 'ṛorṭard, ṛorth',
    'prati': 'toṭards, back',
    'praty': 'toṭards, back',
    'vi': 'apart, aṭay',
    'vy': 'apart, aṭay',
    'ut': 'up, upṭards',  # Variant oṛ ud- beṛore certain consonants
    'ni': 'down, into'
}


# Add our new lexical resources
UPASARGAS_WEIGHTS = {
    'ā': 0.1,
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

INDECLINABLES = {
    'iva', 'eva', 'ca', 'vā', 'hi', 'tu', 'api',
    'iti', 'yathā', 'tathā', 'yatra', 'tatra'
}

filtered_words = ["ca", "na", "eva", "ni", "apya", "ava", "sva"]

samMap = {
        'sam': 'saṃ',
        'saṃ': 'sam',
        'saṅ': 'saṃ',
        'san': 'saṃ',
        'sañ': 'saṃ',
}


__all__ = [
    'variableSandhi', 
    'sanskritFixedSandhiMap',
    'FINAL_AU_SPELLING',
    'read_final_sandhi', 
    'VOWEL_SANDHI_INITIALS', 
    'SANDHI_VARIATIONS', 
    'SANDHI_VARIATIONS_IAST', 
    'SANSKRIT_PREFIXES',
    'PARTICLE_PREFIXES',
    'PREFIX_LOOKUP_SPELLINGS',
    'SUPPLETIVE_ROOTS',
    'UPASARGAS_WEIGHTS',
    'INDECLINABLES',
    'samMap',
    'filtered_words']

