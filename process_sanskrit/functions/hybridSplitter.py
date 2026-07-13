from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from process_sanskrit.functions.sandhiSplitter import sandhi_splitter
from process_sanskrit.functions.compoundAnalysis import root_compounds, process_root_result
from process_sanskrit.functions.sandhiSplitScorer import scorer


@dataclass
class HybridAnalysis:
    """Internal structured result used by caching and public adapters."""

    split: List[str]
    score: float
    subscores: Dict
    source: str
    status: str
    all_splits: Optional[List] = None


def analyze_hybrid(
    text_to_split: str,
    attempts: int = 20,
    score_threshold: float = 0.535,
    *,
    include_candidates: bool = False,
    session=None,
    _memo=None,
) -> HybridAnalysis:
    """Compute one hybrid analysis without consulting the persistent cache."""
    if include_candidates:
        stat_split, stat_score, stat_subscores, all_splits = sandhi_splitter(
            text_to_split,
            cached=False,
            attempts=attempts,
            detailed_output=True,
        )
        if len(stat_split) == 1:
            stat_score = 0
    else:
        stat_split = sandhi_splitter(
            text_to_split,
            cached=False,
            attempts=attempts,
            detailed_output=False,
        )
        all_splits = None
        if len(stat_split) == 1:
            stat_score = 0
            stat_subscores = {}
        else:
            stat_score, stat_subscores = scorer.score_split(
                text_to_split, stat_split
            )

    statistical = HybridAnalysis(
        split=stat_split,
        score=stat_score,
        subscores=stat_subscores,
        source="statistical",
        status="success" if len(stat_split) > 1 else "fallback",
        all_splits=all_splits,
    )
    if stat_score >= score_threshold:
        return statistical

    try:
        root_analysis = root_compounds(
            text_to_split,
            inflection=False,
            session=session,
            _memo=_memo,
        )
        if root_analysis:
            root_split = [process_root_result(item) for item in root_analysis]
            root_split = [
                value
                for index, value in enumerate(root_split)
                if index == 0 or value != root_split[index - 1]
            ]
            root_score, root_subscores = scorer.score_split(
                text_to_split, root_split
            )
            if root_score > stat_score:
                candidates = all_splits
                if include_candidates:
                    candidates = [(root_split, root_score, root_subscores)] + (
                        all_splits if all_splits else []
                    )
                return HybridAnalysis(
                    split=root_split,
                    score=root_score,
                    subscores=root_subscores,
                    source="root_compound",
                    status="success",
                    all_splits=candidates,
                )
    except Exception as error:
        print(f"Root compound analysis failed: {str(error)}")
        statistical.status = "root_analysis_failed"
    return statistical


def hybrid_sandhi_splitter(
    text_to_split: str,
    cached: bool = False,
    attempts: int = 20,
    detailed_output: bool = False,
    score_threshold: float = 0.535,
    session=None,
    _memo=None,
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
            analysis_kind="hybrid",
            algorithm_signature=ANALYSIS_ALGORITHM_VERSION,
            lexicon_fingerprint=lexicon_fingerprint(),
            settings={
                "attempts": attempts,
                "score_threshold": score_threshold,
            },
        )
        record = cache.get(key)
        if record is not None:
            return list(record.split)
        with cache.compute_lock(key) as acquired:
            if acquired:
                record = cache.get(key)
                if record is not None:
                    return list(record.split)
                analysis = analyze_hybrid(
                    text_to_split,
                    attempts,
                    score_threshold,
                    session=session,
                    _memo=_memo,
                )
                canonical = cache.store(
                    CacheRecord(
                        key=key,
                        raw_input=text_to_split,
                        split=analysis.split,
                        score=analysis.score,
                        subscores=analysis.subscores,
                        result_source=analysis.source,
                        status=analysis.status,
                    )
                )
                return list(canonical.split)

    analysis = analyze_hybrid(
        text_to_split,
        attempts,
        score_threshold,
        include_candidates=detailed_output,
        session=session,
        _memo=_memo,
    )
    if detailed_output:
        print("stat_split", analysis.split)
        return (
            analysis.split,
            analysis.score,
            analysis.subscores,
            analysis.all_splits,
        )
    return analysis.split
