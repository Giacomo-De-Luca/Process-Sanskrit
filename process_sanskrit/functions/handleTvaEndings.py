

@dataclass
class TvaAnalysis:
    """
    Represents how to handle a tva-suffixed form by specifying what ending
    needs to be attached to the base word for analysis.
    
    Attributes:
        ending: The ending to attach to the base word after removing tva part
    """
    ending: str

def handle_tva(word: str, session=None) -> Optional[List]:
    """
    Analyzes words containing the -tva suffix by reconstructing the base word
    with appropriate endings for root analysis.
    
    Example:
        For śūnyatvānām:
        1. Identifies -tvānām ending
        2. Gets base śūnya
        3. Analyzes śūnyānām through root_any_word
    """
    # Dictionary mapping tva forms to their analysis ending
    tva_paradigm = {
        # Singular
        'tvam': TvaAnalysis('am'),      # nom/acc: śūnyam
        'tvena': TvaAnalysis('ena'),    # inst: śūnyena 
        'tvāya': TvaAnalysis('āya'),    # dat: śūnyāya
        'tvāt': TvaAnalysis('āt'),      # abl: śūnyāt
        'tvasya': TvaAnalysis('asya'),  # gen: śūnyasya
        'tve': TvaAnalysis('e'),        # loc: śūnye
        
        # Dual
        'tvābhyām': TvaAnalysis('ābhyām'),  # śūnyābhyām
        'tvayoḥ': TvaAnalysis('ayoḥ'),      # śūnyayoḥ
        
        # Plural
        'tvāni': TvaAnalysis('āni'),      # śūnyāni
        'tvebhyaḥ': TvaAnalysis('ebhyaḥ'), # śūnyebhyaḥ
        'tvānām': TvaAnalysis('ānām'),     # śūnyānām
        'tveṣu': TvaAnalysis('eṣu'),       # śūnyeṣu
        
        # Base forms 
        'tva': TvaAnalysis(''),     # Just analyze base
        'tvā': TvaAnalysis('')      # Just analyze base
    }

    if word == "tva":
        return None

    # Try to match a tva ending
    for tva_form, analysis in sorted(tva_paradigm.items(), key=lambda x: len(x[0]), reverse=True):
        if word.endswith(tva_form):
            # Get the base by removing tva form
            base = word[:-len(tva_form)]
            
            #print(f"Found tva form: {tva_form}, base: {base}")
            if base[-1] == 'a':

                # Create the form to analyze by adding the appropriate ending
                analysis_form = base[:-1] + analysis.ending

                #print(f"Reconstructed form: {analysis_form}")
                
                # Analyze this reconstructed form
                base_analysis = root_any_word(analysis_form, session=session)

                #print(f"Base analysis: {base_analysis}")
            
            else: 
                base_analysis = root_any_word(base, session=session)
                ## here it should replace the case ending with the corrisponding case ending of the tva form
                ## base analysis[2] = list of tuples for cases:  [('Nom', 'Sg'), ('Acc', 'Sg')],
                ## 
            
            if base_analysis:
                # Modify results to show tva derivation
                for entry in base_analysis:
                    if isinstance(entry, list) and len(entry) >= 5:
                        entry[1] = f"{entry[1]} + tva"  # Mark as tva derivative
                        entry[4] = word  # Original form
                return base_analysis + ["tva"]
            
    return None