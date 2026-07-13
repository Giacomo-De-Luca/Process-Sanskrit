from .sandhiSplitScorer import scorer
# utils/sandhi_splitter.py
from typing import List, Tuple, Dict, Union, Optional
import ast
import threading
from dataclasses import dataclass


_parser = None
_parser_lock = threading.Lock()


def _get_parser():
    """Construct the comparatively heavy Sanskrit parser on first use."""
    global _parser
    if _parser is None:
        with _parser_lock:
            if _parser is None:
                from ..splitter import Parser

                _parser = Parser(output_encoding='iast')
    return _parser

@dataclass
class SplitResult:
    """Class to hold the result of a sandhi split with scoring information"""
    split: List[str]
    score: float
    subscores: dict
    all_splits: Optional[List[Tuple[List[str], float, dict]]] = None


def analyze_sandhi(
    text_to_split: str,
    attempts: int = 10,
) -> SplitResult:
    """Compute one statistical split and its score without cache I/O."""
    try:
        splits = _get_parser().split(text_to_split, limit=attempts)
        if splits is None:
            simple_split = text_to_split.split()
            if simple_split:
                score, subscores = scorer.score_split(text_to_split, simple_split)
            else:
                score, subscores = 0.0, {}
            return SplitResult(simple_split, score, subscores, None)

        if attempts == 1:
            parsed_splits = [ast.literal_eval(f"{next(splits)}")]
        else:
            parsed_splits = [ast.literal_eval(f"{split}") for split in splits]
        ranked_splits = scorer.rank_splits(text_to_split, parsed_splits)
        best_split, best_score, subscores = ranked_splits[0]
        return SplitResult(best_split, best_score, subscores, ranked_splits)
    except Exception as error:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {error}")
        simple_split = text_to_split.split()
        if simple_split:
            score, subscores = scorer.score_split(text_to_split, simple_split)
        else:
            score, subscores = 0.0, {}
        return SplitResult(simple_split, score, subscores, None)

def sandhi_splitter(
    text_to_split: str, 
    cached: bool = False, 
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
    if cached and not detailed_output:
        from process_sanskrit.utils.analysisCache import (
            ANALYSIS_ALGORITHM_VERSION,
            CacheKey,
            CacheRecord,
            get_analysis_cache,
            lexicon_fingerprint,
        )

        cache = get_analysis_cache(force_enabled=True)
        key = CacheKey.from_settings(
            normalized_input=text_to_split,
            analysis_kind="statistical",
            algorithm_signature=ANALYSIS_ALGORITHM_VERSION,
            lexicon_fingerprint=lexicon_fingerprint(),
            settings={"attempts": attempts},
        )
        record = cache.get(key)
        if record is not None:
            return list(record.split)
        with cache.compute_lock(key) as acquired:
            if acquired:
                record = cache.get(key)
                if record is not None:
                    return list(record.split)
                analysis = analyze_sandhi(text_to_split, attempts)
                canonical = cache.store(
                    CacheRecord(
                        key=key,
                        raw_input=text_to_split,
                        split=analysis.split,
                        score=analysis.score,
                        subscores=analysis.subscores,
                        result_source="statistical",
                        status=(
                            "success" if len(analysis.split) > 1 else "fallback"
                        ),
                    )
                )
                return list(canonical.split)

    analysis = analyze_sandhi(text_to_split, attempts)
    if detailed_output:
        return (
            analysis.split,
            analysis.score,
            analysis.subscores,
            analysis.all_splits,
        )
    return analysis.split
