

from process_sanskrit import process
import time
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
from testing.testCases import test_cases

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_vowel_endings(root: str) -> str:
    """
    Normalize Sanskrit roots that differ in final vowel length.
    This handles all pairs of short and long vowels in Sanskrit:
    - a/ā (अ/आ)
    - i/ī (इ/ई)
    - u/ū (उ/ऊ)
    - ṛ/ṝ (ऋ/ॠ)
    
    For example:
    - 'pariṇāma'/'pariṇāmā' -> 'pariṇāma'
    - 'muni'/'munī' -> 'muni'
    - 'guru'/'gurū' -> 'guru'
    - 'pitṛ'/'pitṝ' -> 'pitṛ'
    
    Args:
        root: Sanskrit root in IAST transliteration
        
    Returns:
        Normalized form of the root with short vowel endings
    """
    # Dictionary mapping long vowels to their short counterparts
    vowel_pairs = {
        'ā': 'a',
        'ī': 'i',
        'ū': 'u',
        'ṝ': 'ṛ'  # This is rare but can occur in certain grammatical forms
    }
    
    if not root:
        return root
        
    # Check if the word ends with any long vowel
    final_char = root[-1]
    if final_char in vowel_pairs:
        # Replace the final long vowel with its short counterpart
        return root[:-1] + vowel_pairs[final_char]
        
    return root


class SanskritBenchmark:
    """Comprehensive benchmark suite for Sanskrit text processing."""
    
    def __init__(self):
        """Initialize benchmark with test cases covering various Sanskrit constructions."""
        

        self.test_cases = test_cases
        self.results = {}
        self.performance_metrics = {}

    def evaluate_result(self, 
                    result: Dict[str, List[str]],
                    case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a processing result against the test case. The evaluation considers
        a result correct if all expected roots are found in the results, even if
        additional valid derivations are present.
        
        For example, if the expected roots are ['rāma', 'lakṣmaṇa'] and the results
        contain ['rāma', 'rā', 'lakṣmaṇa'], this is considered a 100% match since
        all expected roots are present, even though an additional valid root 'rā'
        was also found.
        
        Args:
            result: The roots returned by the processing function
            case: The test case containing correct splits and possible roots
                
        Returns:
            Dictionary containing evaluation metrics and analysis details
        """
        evaluation = {
            'input': case['input'],
            'predicted': result,
            'expected_roots': case['possible_roots'],
            'scores': {},
            'notes': []
        }

        #print(f"result: {result}")
        
        # Convert result to a set for easier membership testing
        result_set = {key for key in result} | {item for sublist in result.values() for item in sublist}
        
        # For each set of possible roots in the test case
        max_root_score = 0
        best_matching_roots = None
        
        for possible_roots in case['possible_roots']:
            # Convert possible roots to set for comparison

            #print(f"possible_roots: {possible_roots}")
            required_roots = set(possible_roots)
            
            # Check if all required roots are present in the results
            all_roots_found = required_roots.issubset(result_set)
            
            if all_roots_found:
                # If we found all required roots, this is a perfect score
                max_root_score = 1.0
                best_matching_roots = possible_roots
                evaluation['notes'].append("All required roots found")
                # Found a perfect match, no need to check other possibilities
                break
            else:
                # Calculate partial score based on how many required roots were found
                matched_roots = required_roots & result_set
                score = len(matched_roots) / len(required_roots)
                if score > max_root_score:
                    max_root_score = score
                    best_matching_roots = possible_roots
        
        evaluation['scores']['root_accuracy'] = max_root_score
        
        # Add detailed analysis of the match
        if max_root_score < 1.0 and best_matching_roots:
            missing_roots = set(best_matching_roots) - result_set
            extra_roots = result_set - set(best_matching_roots)
            if missing_roots:
                evaluation['notes'].append(f"Missing roots: {', '.join(missing_roots)}")
            if extra_roots:
                evaluation['notes'].append(f"Additional roots found: {', '.join(extra_roots)}")
        
        # Add special case notes
        if len(result) == 1 and result[0] == case['input']:
            evaluation['notes'].append("Returned as dictionary entry")
            
        if case.get('notes'):
            evaluation['notes'].append(f"Test case note: {case['notes']}")
            
        return evaluation
    

    def evaluate_result(self, 
                    result: List[str], 
                    case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate processing results with comprehensive handling of Sanskrit vowel variations.
        This improved version considers all vowel length variations that commonly occur
        in Sanskrit word endings.
        
        For example, these pairs would receive partial credit:
        - deva/devā (देव/देवा)
        - kavi/kavī (कवि/कवी)
        - guru/gurū (गुरु/गुरू)
        - kartṛ/kartṝ (कर्तृ/कर्तॄ)
        
        Args:
            result: The roots returned by the processing function
            case: The test case containing correct splits and possible roots
                
        Returns:
            Dictionary containing evaluation metrics and detailed analysis
        """
        evaluation = {
            'input': case['input'],
            'predicted': result,
            'expected_roots': case['possible_roots'],
            'scores': {},
            'notes': [],
            'vowel_variations': []  # Track which vowel variations were found
        }
        
        # Convert result to a set and create normalized versions
        result_set = {key for key in result} | {item for sublist in result.values() for item in sublist}
        normalized_results = {normalize_vowel_endings(r) for r in result}
        
        max_root_score = 0
        best_matching_roots = None
        
        for possible_roots in case['possible_roots']:
            # Create normalized versions of expected roots
            normalized_expected = {normalize_vowel_endings(r) for r in possible_roots}
            
            # Calculate exact matches
            exact_matches = set(possible_roots) & result_set
            
            # Calculate vowel variation matches (excluding exact matches)
            normalized_matches = normalized_expected & normalized_results
            vowel_variation_matches = set()
            
            for norm_match in normalized_matches:
                # Find the original forms that normalized to this match
                orig_expected = {r for r in possible_roots if normalize_vowel_endings(r) == norm_match}
                orig_results = {r for r in result if normalize_vowel_endings(r) == norm_match}
                
                # If they're different but normalize to the same thing, it's a vowel variation
                if orig_expected != orig_results:
                    vowel_variation_matches.update(orig_results)
                    evaluation['vowel_variations'].append({
                        'expected': list(orig_expected)[0],
                        'found': list(orig_results)[0],
                        'normalized': norm_match
                    })
            
            # Calculate score with partial credit for vowel variations
            score = (len(exact_matches) + 0.5 * len(vowel_variation_matches)) / len(possible_roots)
            
            if score > max_root_score:
                max_root_score = score
                best_matching_roots = possible_roots
                
                if score == 1.0:
                    evaluation['notes'].append("Perfect match found")
                elif vowel_variation_matches:
                    for var in evaluation['vowel_variations']:
                        evaluation['notes'].append(
                            f"Found vowel variation: {var['found']} (expected {var['expected']})"
                        )
        
        evaluation['scores']['root_accuracy'] = max_root_score
        
        # Add detailed analysis
        if best_matching_roots:
            exact_matches = set(best_matching_roots) & result_set
            missing_roots = set(best_matching_roots) - result_set
            extra_roots = result_set - set(best_matching_roots)
            
            if missing_roots:
                evaluation['notes'].append(f"Missing roots: {', '.join(missing_roots)}")
            if extra_roots:
                evaluation['notes'].append(f"Additional roots found: {', '.join(extra_roots)}")
        
        return evaluation

    def run_benchmark(self, 
                    dict_names: List[str] = None, 
                    verbose: bool = False) -> Dict[str, Any]:
        """
        Run the benchmark suite.
        
        Args:
            dict_names: List of dictionary names to use
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary containing benchmark results
        """
        results = {
            'overall': {'total': 0, 'root_accuracy': 0.0},
            'by_type': {},
            'by_complexity': {},
            'timing': [],
            'timing_stats': {'mean': 0, 'min': 0, 'max': 0},  # Initialize timing_stats
            'detailed_results': []
        }
    
        for case in self.test_cases:
            if verbose:
                logger.info(f"Processing: {case['input']}")
                
            try:
                # Check if case has the required keys
                if not all(key in case for key in ['input', 'correct_split', 'type', 'complexity']):
                    logger.warning(f"Skipping invalid test case: {case}")
                    continue

                # Add possible_roots if not present
                if 'possible_roots' not in case:
                    case['possible_roots'] = [case['correct_split']]
                
                # Time the processing
                start_time = time.time()
                processed_result = process(
                    case['input'],
                    dict_names[0] if dict_names else 'MW',  # Default to MW if none specified
                    roots="parts"
                )
                processing_time = time.time() - start_time
                
                # Evaluate the result
                evaluation = self.evaluate_result(processed_result, case)
                evaluation['processing_time'] = processing_time
                evaluation['type'] = case['type']
                evaluation['complexity'] = case['complexity']
                
                results['detailed_results'].append(evaluation)
                
                # Update statistics
                results['overall']['total'] += 1
                results['overall']['root_accuracy'] += evaluation['scores']['root_accuracy']
                
                # Update type statistics
                if case['type'] not in results['by_type']:
                    results['by_type'][case['type']] = {'total': 0, 'root_accuracy': 0.0}
                type_stats = results['by_type'][case['type']]
                type_stats['total'] += 1
                type_stats['root_accuracy'] += evaluation['scores']['root_accuracy']
                
                # Update complexity statistics
                if case['complexity'] not in results['by_complexity']:
                    results['by_complexity'][case['complexity']] = {'total': 0, 'root_accuracy': 0.0}
                complexity_stats = results['by_complexity'][case['complexity']]
                complexity_stats['total'] += 1
                complexity_stats['root_accuracy'] += evaluation['scores']['root_accuracy']
                
                results['timing'].append(processing_time)
                
            except Exception as e:
                logger.error(f"Error processing {case['input']}: {str(e)}")
                continue
                
        # Calculate averages
        if results['overall']['total'] > 0:
            results['overall']['root_accuracy'] /= results['overall']['total']
            
        for category in [results['by_type'], results['by_complexity']]:
            for stats in category.values():
                if stats['total'] > 0:
                    stats['root_accuracy'] /= stats['total']
        
        # Calculate timing statistics if we have timing data
        if results['timing']:
            results['timing_stats'] = {
                'mean': sum(results['timing']) / len(results['timing']),
                'min': min(results['timing']),
                'max': max(results['timing'])
            }
        
        self.results = results
        return results

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a detailed report of benchmark results.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report text
        """
        if not hasattr(self, 'results') or not self.results:
            return "No benchmark results available. Please run the benchmark first."

        report = []
        report.append("=== Sanskrit Processing Benchmark Report ===\n")
        
        # Overall Performance
        report.append("Overall Performance:")
        report.append(f"Total Cases: {self.results['overall']['total']}")
        report.append(
            f"Average Root Accuracy: "
            f"{self.results['overall']['root_accuracy']*100:.2f}%\n"
        )
        
        # Performance by Type
        report.append("Performance by Type:")
        for type_name, stats in self.results['by_type'].items():
            report.append(f"  {type_name}:")
            report.append(f"    Cases: {stats['total']}")
            report.append(
                f"    Root Accuracy: {stats['root_accuracy']*100:.2f}%"
            )
        report.append("")
        
        # Performance by Complexity
        report.append("Performance by Complexity:")
        for complexity, stats in self.results['by_complexity'].items():
            report.append(f"  {complexity}:")
            report.append(f"    Cases: {stats['total']}")
            report.append(
                f"    Root Accuracy: {stats['root_accuracy']*100:.2f}%"
            )
        report.append("")
        
        # Timing Statistics (with safety checks)
        if 'timing_stats' in self.results and self.results['timing']:
            report.append("Processing Time Statistics:")
            report.append(
                f"  Mean: {self.results['timing_stats']['mean']*1000:.2f}ms"
            )
            report.append(
                f"  Min: {self.results['timing_stats']['min']*1000:.2f}ms"
            )
            report.append(
                f"  Max: {self.results['timing_stats']['max']*1000:.2f}ms\n"
            )
        
        # Detailed Results
        report.append("Detailed Results:")
        for result in self.results['detailed_results']:
            report.append(f"  Input: {result['input']}")
            report.append(f"    Predicted: {result['predicted']}")
            report.append(f"    Expected: {result.get('expected_roots', 'Not specified')}")
            report.append(
                f"    Root Accuracy: "
                f"{result['scores']['root_accuracy']*100:.2f}%"
            )
            if result.get('notes'):
                report.append(f"    Notes: {'; '.join(result['notes'])}\n")
        
        report_text = "\n".join(report)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
            except Exception as e:
                logger.error(f"Error writing report to file: {str(e)}")
        
        return report_text

if __name__ == "__main__":
    # Example usage
    benchmark = SanskritBenchmark()
    results = benchmark.run_benchmark(dict_names=['MW'], verbose=True)
    report = benchmark.generate_report('sanskrit_benchmark_results.txt')
    print(report)