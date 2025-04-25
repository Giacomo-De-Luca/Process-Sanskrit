
from typing import List, Tuple, Dict, Union
from process_sanskrit.functions.enhancedSandhiSplitter import enhanced_sandhi_splitter
from process_sanskrit.functions.compoundAnalysis import root_compounds, process_root_result
from process_sanskrit.functions.sandhiSplitScorer import scorer


def hybrid_sandhi_splitter(
    text_to_split: str,
    cached: bool = False,
    attempts: int = 20,
    detailed_output: bool = False,
    score_threshold: float = 0.535,
    count_types: bool = False
) -> Union[List[str], Tuple[List[str], float, Dict, List], Tuple[List[str], Dict]]:
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

     # Initialize counts dictionary if tracking
    compound_calls = 0 if count_types else None
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
        if detailed_output:
            print("stat_score", stat_score)
            print("stat_split", stat_split)
        if count_types:
            return compound_calls
        if detailed_output:
            return stat_split, stat_score, stat_subscores, all_splits
        return stat_split

    # If score is too low, try root compound analysis
    try:
        if count_types:
            compound_calls += 1
        if detailed_output == True:
            print("text_to_split", text_to_split)
        root_analysis = root_compounds(text_to_split, inflection=False)
        if detailed_output == True:
            print("root_analysis", root_analysis)
        if root_analysis:
            # Process the root analysis results into a simple word list
            root_split = [process_root_result(item) for item in root_analysis]
            root_split = [x for i, x in enumerate(root_split) if i == 0 or x != root_split[i-1]]
            if detailed_output == True:
                print("root_split", root_split)
            # Score the processed root split
            root_score, root_subscores = scorer.score_split(text_to_split, root_split)
            if detailed_output == True:
                print("root_score", root_score)
                print("root_subscores", root_subscores)

            # Compare scores and choose the better result
            if root_score > stat_score:
                if count_types:
                    return compound_calls
                if detailed_output:
                    # Add root split to all_splits for reference
                    all_splits = [(root_split, root_score, root_subscores)] + (all_splits if all_splits else [])
                    return root_split, root_score, root_subscores, all_splits
                return root_split

    except Exception as e:
        print(f"Root compound analysis failed: {str(e)}")

    if count_types:
        return compound_calls

    # Fall back to statistical split if root analysis fails or scores lower
    if detailed_output:
        print("stat_split", stat_split)
        return stat_split, stat_score, stat_subscores, all_splits

    return stat_split


