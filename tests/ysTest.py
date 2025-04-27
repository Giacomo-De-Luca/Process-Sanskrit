from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import json
import datetime
from tests.datasets.yogaSutra import ys
from process_sanskrit import process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YogaSutraAnalyzer:
    """
    A comprehensive analyzer for processing and evaluating Sanskrit segmentation
    on the Yoga Sutras. This class implements both full-line and word-by-word
    analysis, with facilities for expert validation and comparison.
    """
    
    def __init__(self, sutras: List[str]):
        """
        Initialize the analyzer with a list of Yoga Sutra lines.
        
        Args:
            sutras: List of Sanskrit strings containing the Yoga Sutras
        """
        self.sutras = sutras
        self.results = {
            'full_line': [],    # Results from processing entire lines
            'word_by_word': [], # Results from processing individual words
            'expert_validations': {}  # Store expert validations when available
        }
        
    def process_full_lines(self, dict_names: List[str] = None) -> List[Dict]:
        """
        Process each sutra line as a complete unit.
        
        Args:
            dict_names: List of dictionary names to use for processing
            
        Returns:
            List of processing results for each line
        """
        logger.info("Processing full sutra lines...")
        
        for i, sutra in enumerate(self.sutras, 1):
            try:
                # Process the full line
                result = process(sutra, dict_names[0] if dict_names else 'MW', roots="parts")
                
                # Store the result with metadata
                analysis = {
                    'sutra_number': i,
                    'original_text': sutra,
                    'segmentation': result,
                    'processing_level': 'full_line'
                }
                
                self.results['full_line'].append(analysis)
                logger.info(f"Processed sutra {i}: Found {len(result)} segments")
                
            except Exception as e:
                logger.error(f"Error processing sutra {i}: {str(e)}")
                
        return self.results['full_line']
    
    def process_word_by_word(self, dict_names: List[str] = None) -> List[Dict]:
        """
        Process each word in each sutra separately.
        
        Args:
            dict_names: List of dictionary names to use for processing
            
        Returns:
            List of processing results for each word
        """
        logger.info("Processing individual words...")
        
        for i, sutra in enumerate(self.sutras, 1):
            try:
                # Split the line into words (this is simplified - might need more sophisticated splitting)
                words = sutra.split()
                
                word_results = []
                for word in words:
                    # Process each word
                    result = process(word, dict_names[0] if dict_names else 'MW', roots="parts")
                    word_results.append({
                        'original_word': word,
                        'segmentation': result
                    })
                
                # Store results with metadata
                analysis = {
                    'sutra_number': i,
                    'original_text': sutra,
                    'word_analysis': word_results,
                    'processing_level': 'word_by_word'
                }
                
                self.results['word_by_word'].append(analysis)
                logger.info(f"Processed sutra {i}: Analyzed {len(words)} words")
                
            except Exception as e:
                logger.error(f"Error processing words in sutra {i}: {str(e)}")
                
        return self.results['word_by_word']
    
    def add_expert_validation(self, 
                            sutra_number: int, 
                            validation: Dict[str, Any],
                            expert_id: str = None) -> None:
        """
        Add expert validation for a specific sutra.
        
        Args:
            sutra_number: The number of the sutra being validated
            validation: Dictionary containing expert's analysis
            expert_id: Optional identifier for the expert
        """
        if sutra_number not in self.results['expert_validations']:
            self.results['expert_validations'][sutra_number] = []
            
        validation['expert_id'] = expert_id
        validation['timestamp'] = datetime.datetime.now()

        
        self.results['expert_validations'][sutra_number].append(validation)
    
    def compare_results(self) -> Dict[str, Any]:
        """
        Compare full-line and word-by-word processing results.
        When available, also compare against expert validations.
        
        Returns:
            Dictionary containing comparison metrics and analysis
        """
        comparison = {
            'total_sutras': len(self.sutras),
            'processing_comparison': [],
            'expert_validated_count': len(self.results['expert_validations'])
        }
        
        for i, sutra in enumerate(self.sutras, 1):
            sutra_comparison = {
                'sutra_number': i,
                'original_text': sutra
            }
            
            # Get both types of processing results
            full_line = next((r for r in self.results['full_line'] 
                            if r['sutra_number'] == i), None)
            word_by_word = next((r for r in self.results['word_by_word']
                               if r['sutra_number'] == i), None)
            
            if full_line and word_by_word:
                # Compare the results
                sutra_comparison['full_line_segments'] = len(full_line['segmentation'])
                sutra_comparison['word_by_word_segments'] = sum(
                    len(w['segmentation']) for w in word_by_word['word_analysis']
                )
                
                # Add expert validation if available
                if i in self.results['expert_validations']:
                    sutra_comparison['expert_validated'] = True
                    sutra_comparison['expert_analysis'] = self.results['expert_validations'][i]
                
            comparison['processing_comparison'].append(sutra_comparison)
        
        return comparison
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a comprehensive analysis report showing both full-line and word-by-word
        processing results for each sutra. This enhanced report includes the actual
        segmentation results to enable detailed analysis of the system's performance.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Formatted report text with detailed analysis of each sutra
        """
        report = []
        report.append("=== Yoga Sutra Processing Analysis ===\n")
        
        # Add overall statistics
        comparison = self.compare_results()
        report.append(f"Total Sutras Analyzed: {comparison['total_sutras']}")
        report.append(
            f"Sutras with Expert Validation: {comparison['expert_validated_count']}\n"
        )
        
        # Add detailed analysis for each sutra
        report.append("Detailed Analysis by Sutra:")
        
        # For each sutra, we'll show both processing approaches
        for i, sutra in enumerate(self.sutras, 1):
            report.append(f"\n{'='*80}")  # Visual separator between sutras
            report.append(f"\nSutra {i}:")
            report.append(f"Original Text: {sutra}")
            
            # Get full-line results for this sutra
            full_line_result = next(
                (r for r in self.results['full_line'] if r['sutra_number'] == i), 
                None
            )
            if full_line_result:
                report.append("\nFull Line Analysis:")
                report.append("------------------")
                # Format the segmentation results
                if isinstance(full_line_result['segmentation'], dict):
                    for compound, parts in full_line_result['segmentation'].items():
                        report.append(f"  {compound} → {parts}")
                else:
                    report.append(f"  Segmentation: {full_line_result['segmentation']}")
            
            # Get word-by-word results for this sutra
            word_result = next(
                (r for r in self.results['word_by_word'] if r['sutra_number'] == i),
                None
            )
            if word_result:
                report.append("\nWord-by-Word Analysis:")
                report.append("---------------------")
                for word_analysis in word_result['word_analysis']:
                    original = word_analysis['original_word']
                    segmentation = word_analysis['segmentation']
                    if isinstance(segmentation, dict):
                        report.append(f"  {original}:")
                        for compound, parts in segmentation.items():
                            report.append(f"    {compound} → {parts}")
                    else:
                        report.append(f"  {original} → {segmentation}")
            
            # Add expert validation if available
            if i in self.results['expert_validations']:
                report.append("\nExpert Validation:")
                report.append("-----------------")
                for validation in self.results['expert_validations'][i]:
                    if validation.get('expert_id'):
                        report.append(f"Expert: {validation['expert_id']}")
                    report.append(f"Correct segments: {validation['correct_segments']}")
                    if validation.get('notes'):
                        report.append(f"Notes: {validation['notes']}")
                    report.append(f"Validated at: {validation['timestamp']}\n")
            
            # Add comparison metrics
            comparison_entry = next(
                (c for c in comparison['processing_comparison'] 
                if c['sutra_number'] == i),
                None
            )
            if comparison_entry:
                report.append("\nComparison Metrics:")
                report.append("------------------")
                if 'full_line_segments' in comparison_entry:
                    report.append(
                        f"Full-line total segments: {comparison_entry['full_line_segments']}"
                    )
                    report.append(
                        f"Word-by-word total segments: "
                        f"{comparison_entry['word_by_word_segments']}"
                    )
                # Add analysis of differences between approaches
                if ('full_line_segments' in comparison_entry and 
                    'word_by_word_segments' in comparison_entry):
                    diff = abs(comparison_entry['full_line_segments'] - 
                            comparison_entry['word_by_word_segments'])
                    if diff > 0:
                        report.append(
                            f"Note: The two approaches differ by {diff} segments"
                        )
        
        report_text = "\n".join(report)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Report successfully written to {output_path}")
            except Exception as e:
                logger.error(f"Error writing report to file: {str(e)}")
        
        return report_text

# Example usage
if __name__ == "__main__":
    # Sample usage with some Yoga Sutras
    yoga_sutras = ys
    
    analyzer = YogaSutraAnalyzer(yoga_sutras)
    
    # Process both ways
    analyzer.process_full_lines(['MW'])
    analyzer.process_word_by_word(['MW'])
    
    # Add some expert validation
    analyzer.add_expert_validation(1, {
        'correct_segments': ['atha', 'yoga', 'anuśāsanam'],
        'notes': 'Clear case of prefix anu- with śās'
    })
    
    # Generate and print report
    report = analyzer.generate_report('yoga_sutra_analysis.txt')
    print(report)