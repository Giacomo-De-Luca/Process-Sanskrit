#!/usr/bin/env python3
"""
Script to process Sanskrit compounds from JSON file and output results for manual evaluation.
"""

import json
import logging
from datetime import datetime
import os
from process_sanskrit import process

# Set up logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_compounds(
    json_path: str = "tests/datasets/sanskrit_compounds_benchmark.json",
    output_path: str = "tests/results/compound_processing_results.txt"
):
    """
    Process all compounds from the JSON file and print results.
    """
    # Load compounds from JSON
    print(f"Loading compounds from {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            compounds = data['compounds']
    except FileNotFoundError:
        print(f"Could not find file at {json_path}")
        print(f"Current working directory: {os.getcwd()}")
        return
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Process each category
        for category in ['long', 'medium', 'short']:
            header = f"\n{category.upper()} COMPOUNDS\n{'='*80}\n"
            print(header)
            outfile.write(header)

            for idx, case in enumerate(compounds[category], 1):
                text = case['text']
                source = case['source_file']
                
                entry = f"\nCompound #{idx}\n"
                entry += f"Source: {source}\n"
                entry += f"Input: {text}\n"
                
                try:
                    result = process(text, roots="roots")
                    entry += f"Result: {result}\n"
                except Exception as e:
                    error = f"Error processing compound: {str(e)}\n"
                    entry += error
                
                entry += "-"*80 + "\n"
                
                print(entry)
                outfile.write(entry)

        print(f"\nResults have been saved to {output_path}")

if __name__ == "__main__":
    process_compounds()