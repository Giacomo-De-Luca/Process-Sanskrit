#!/usr/bin/env python3
"""
Main script to run Sanskrit processing benchmarks and Yoga Sutra analysis.
This script provides a unified entry point for various Sanskrit text processing tests,
including both standardized benchmarks and analysis of the Yoga Sutras.
"""

import os
import sys
from datetime import datetime
import argparse
import json

from processSanskrit import process
from tests.benchmarkTesting import SanskritBenchmark
from tests.datasets.yogaSutra import ys

def run_standard_benchmarks():
    """
    Executes the standard benchmark suite using the SanskritBenchmark class.
    This maintains the original benchmark functionality for testing basic Sanskrit
    processing capabilities.
    """
    try:
        
        
        print("\nRunning Standard Benchmarks:")
        print("-" * 40)
        
        benchmark = SanskritBenchmark()
        results = benchmark.run_benchmark(dict_names=['MW'], verbose=True)
        report = benchmark.generate_report('sanskrit_benchmark_results.txt')
        
        print("\nBenchmark Results:")
        print("-" * 20)
        print(report)
        
        return True
        
    except Exception as e:
        print(f"Error in standard benchmarks: {e}")
        return False
    




def run_yoga_sutra_analysis():
    """
    Executes the Yoga Sutra analysis using the YogaSutraAnalyzer class.
    This provides detailed analysis of how the system handles actual Sanskrit texts
    from the Yoga Sutras.
    """
    try:
        from testing.ysTest import YogaSutraAnalyzer
        
        print("\nRunning Yoga Sutra Analysis:")
        print("-" * 40)
        
        analyzer = YogaSutraAnalyzer(ys)
        
        # Run both types of analysis
        analyzer.process_full_lines(['MW'])
        analyzer.process_word_by_word(['MW'])
        
        # Generate and save the analysis report
        report = analyzer.generate_report('yoga_sutra_analysis.txt')
        print(report)
        
        return True
        
    except Exception as e:
        print(f"Error in Yoga Sutra analysis: {e}")
        return False
def group_entries(data):
    """Group word entries by etymology and extract roots."""
    grouped_entries = {}
    
    # Handle different return types from process()
    if not data:  # Handle None or empty list
        return grouped_entries, set()
        
    if isinstance(data, str):  # Handle string returns
        grouped_entries['default'] = [data]
        return grouped_entries, {data}
        
    # First collect roots by etymology
    roots = set()
    for entry in data:
        if isinstance(entry, list):
            if len(entry) >= 7:  # Long entry format
                key = entry[4] if entry[4] else 'default'
                roots.add(entry[0])  # Add the root form
            elif len(entry) >= 3:  # Short entry format
                key = 'default'
                roots.add(entry[0])
            else:
                key = 'default'
                
            if key not in grouped_entries:
                grouped_entries[key] = []
            grouped_entries[key].append(entry)
        elif isinstance(entry, str):
            if 'default' not in grouped_entries:
                grouped_entries['default'] = []
            grouped_entries['default'].append(entry)
            roots.add(entry)
            
    return grouped_entries, roots

def format_compound_entry(text: str, source: str, idx: int, result) -> str:
    """Format a single compound's analysis in a clean, readable way."""
    grouped, roots = group_entries(result)
    
    # Start with header
    entry = []
    entry.append(f"{'='*80}")
    entry.append(f"Compound #{idx}")
    entry.append(f"Source: {source}")
    entry.append(f"Input: {text}")
    entry.append("-" * 40)
    
    # Add roots summary
    num_segments = len(roots)
    num_groups = len(grouped)
    if num_segments > 0:
        entry.append(f"Found {num_segments} root{'s' if num_segments > 1 else ''}:")
        entry.append(f"Found {num_groups} group{'s' if num_groups > 1 else ''}:")
        entry.append(f"  {', '.join(sorted(roots))}")
        entry.append("")
        
        # Add grouped results with cleaner formatting
        entry.append("Analysis by found words:")
        entry.append("-" * 20)
        
        # Sort groups, putting 'default' last if it exists
        group_keys = sorted(grouped.keys())
            
        for group_key in group_keys:
            entries = grouped[group_key]
            # Only show groups with actual content
            roots_in_group = {e[0] if isinstance(e, list) and len(e) >= 1 else e 
                            for e in entries}
            if roots_in_group:
                group_name = "" if group_key == "default" else f" [{group_key}]"
                entry.append(f"• Found form{group_name}:")
                for root in sorted(roots_in_group):
                    entry.append(f"    - {root}")
                entry.append("")
    else:
        entry.append("No roots found.")
        
    return "\n".join(entry)

def run_compound_processing():
    """Process compounds from JSON file with improved formatting."""
    print("\nProcessing compounds...")
    category_stats = {
        'long': {'total_compounds': 0, 'total_segments': 0, 'total_groups': 0,  'successful_parses': 0},
        'medium': {'total_compounds': 0, 'total_segments': 0, 'total_groups': 0,  'successful_parses': 0},
        'short': {'total_compounds': 0, 'total_segments': 0, 'total_groups': 0,  'successful_parses': 0},
        'very_short': {'total_compounds': 0, 'total_segments': 0, 'total_groups': 0,  'successful_parses': 0},
        
    }
    
    try:
        # Load compounds from JSON
        with open("testing/sanskrit_compounds_benchmark.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            compounds = data['compounds']
        
        # Create output file
        with open("testing/compound_processing_results.txt", 'w', encoding='utf-8') as outfile:
            for category in ['long', 'medium', 'short', 'very_short']:
                header = f"\n{category.upper()} COMPOUNDS\n{'='*80}\n"
                print(header)
                outfile.write(header)
                
                category_stats[category]['total_compounds'] = len(compounds[category])

                for idx, case in enumerate(compounds[category], 1):
                    try:
                        # Process the compound
                        result = process(case['text'])
                        
                        # Format the entry
                        entry = format_compound_entry(
                            case['text'],
                            case['source_file'],
                            idx,
                            result
                        )
                        
                        # Update statistics
                        grouped, roots = group_entries(result)
                        num_segments = len(roots)
                        num_groups = len(grouped)
                        
                        
                        if num_segments > 0:
                            category_stats[category]['successful_parses'] += 1
                            category_stats[category]['total_segments'] += num_segments
                            category_stats[category]['total_groups'] += num_groups

                            
                    except Exception as e:
                        entry = f"\nError processing compound {idx}: {str(e)}\n"
                        entry += f"Source: {case['source_file']}\n"
                        entry += f"Input: {case['text']}\n"
                        entry += "-"*80 + "\n"
                    
                    print(entry)
                    outfile.write(entry + "\n")

                # Add category summary with improved formatting
                summary = [
                    f"\n{category.upper()} CATEGORY SUMMARY",
                    "=" * 40,
                    f"Total compounds processed: {category_stats[category]['total_compounds']}",
                    f"Successfully parsed: {category_stats[category]['successful_parses']}",
                    f"Total roots found: {category_stats[category]['total_segments']}",
                    f"Total groups found: {category_stats[category]['total_groups']}",

                ]
                
                if category_stats[category]['successful_parses'] > 0:
                    avg = (category_stats[category]['total_segments'] / 
                          category_stats[category]['successful_parses'])
                    avgGroups = (category_stats[category]['total_groups'] / 
                          category_stats[category]['successful_parses'])
                    success_rate = (category_stats[category]['successful_parses'] / 
                                  category_stats[category]['total_compounds'] * 100)
                    summary.extend([
                        f"Success rate: {success_rate:.1f}%",
                        f"Average roots per compound: {avg:.2f}"
                        f"Average roots per group: {avgGroups:.2f}"

                    ])
                    
                summary = "\n".join(summary) + "\n" + "=" * 40 + "\n"
                print(summary)
                outfile.write(summary)

            # Add overall summary with improved formatting
            total_compounds = sum(stats['total_compounds'] for stats in category_stats.values())
            total_successful = sum(stats['successful_parses'] for stats in category_stats.values())
            total_roots = sum(stats['total_segments'] for stats in category_stats.values())
            total_groups = sum(stats['total_groups'] for stats in category_stats.values())
            
            overall = [
                "\nOVERALL ANALYSIS",
                "=" * 40,
                f"Total compounds analyzed: {total_compounds}",
                f"Successfully parsed: {total_successful}",
                f"Total roots found: {total_roots}"
                f"Total groups found: {total_groups}"
            ]
            
            if total_successful > 0:
                success_rate = total_successful / total_compounds * 100
                avg_roots = total_roots / total_successful
                overall.extend([
                    f"Overall success rate: {success_rate:.1f}%",
                    f"Average roots per compound: {avg_roots:.2f}",
                    f"Average groups per compounds: {total_groups / total_successful:.2f}"
                ])
                
            overall = "\n".join(overall) + "\n" + "=" * 40
            print(overall)
            outfile.write(overall)
            
            print(f"\nResults have been saved to testing/compound_processing_results.txt")
            
        return True
        
    except Exception as e:
        print(f"Error processing compounds: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def main():
    """
    Main entry point for running Sanskrit text processing tests.
    Handles command line arguments to determine which tests to run
    and sets up the environment appropriately.
    """
    parser = argparse.ArgumentParser(
        description='Run Sanskrit processing tests and analysis'
    )
    parser.add_argument(
        '--benchmarks', 
        action='store_true',
        help='Run standard benchmark tests'
    )
    parser.add_argument(
        '--yoga-sutras', 
        action='store_true',
        help='Run Yoga Sutra analysis'
    )
    parser.add_argument(
        '--compounds',
        action='store_true',
        help='Run compound analysis benchmarks'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all available tests'
    )
    
    args = parser.parse_args()
    
    # If no specific tests are requested, run all
    if not (args.benchmarks or args.yoga_sutras or args.compounds):
        args.all = True
    
    print(f"Starting Sanskrit analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    success = True
    
    # Run requested tests
    if args.all or args.benchmarks:
        success = run_standard_benchmarks() and success
        
    if args.all or args.yoga_sutras:
        success = run_yoga_sutra_analysis() and success
        
    if args.all or args.compounds:
        success = run_compound_processing() and success
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()


 