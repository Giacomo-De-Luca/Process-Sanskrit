"""
Test suite generator for Sanskrit compounds from GRETIL corpus.
Integrates with existing benchmark infrastructure.
"""
from typing import List, Dict, Any
import random
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class GretilTestSuite:
    """
    Manages test cases generated from GRETIL corpus compounds.
    Integrates with existing benchmark system.
    """
    
    def __init__(self, compounds_file: str):
        """
        Initialize test suite from compounds file.
        
        Args:
            compounds_file: Path to JSON file containing extracted compounds
        """
        self.compounds_file = Path(compounds_file)
        self.test_cases = []
        self._load_compounds()
        
    def _load_compounds(self) -> None:
        """Load and validate compounds from JSON file."""
        try:
            with open(self.compounds_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.compounds = data['compounds']
                self.metadata = data['metadata']
        except Exception as e:
            logger.error(f"Error loading compounds file: {str(e)}")
            raise
            
    def generate_test_cases(self, 
                          samples_per_category: int = 20,
                          include_source: bool = True) -> List[Dict[str, Any]]:
        """
        Generate test cases in format compatible with benchmark system.
        
        Args:
            samples_per_category: Number of samples to use per length category
            include_source: Whether to include source file information
            
        Returns:
            List of test cases in benchmark-compatible format
        """
        test_cases = []
        
        for category, compounds in self.compounds.items():
            # Sample compounds for this category
            selected = random.sample(
                compounds,
                min(samples_per_category, len(compounds))
            )
            
            for compound in selected:
                test_case = {
                    "input": compound['text'],
                    "type": "compound",
                    "complexity": category,
                    # For compounds, we initially set an empty list for possible_roots
                    # These can be manually validated later
                    "possible_roots": [],
                    "notes": f"Extracted from GRETIL corpus"
                }
                
                if include_source:
                    test_case["source_file"] = compound['source_file']
                    
                test_cases.append(test_case)
                
        return test_cases
        
    def save_test_suite(self, 
                       output_file: str,
                       samples_per_category: int = 20) -> None:
        """
        Save generated test suite to file.
        
        Args:
            output_file: Path to save test suite
            samples_per_category: Number of samples per category
        """
        test_cases = self.generate_test_cases(samples_per_category)
        
        output = {
            "metadata": {
                "source": "GRETIL corpus",
                "samples_per_category": samples_per_category,
                "categories": self.metadata['categories'],
                "total_available": {
                    category: len(compounds)
                    for category, compounds in self.compounds.items()
                }
            },
            "test_cases": test_cases
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        logger.info(
            f"Saved {len(test_cases)} test cases to {output_file}"
        )

def get_gretil_test_cases() -> List[Dict[str, Any]]:
    """
    Factory function to get GRETIL test cases for benchmark system.
    Returns test cases in format compatible with existing benchmarks.
    """
    # Use Path for cross-platform compatibility and relative path resolution
    compounds_file = Path(__file__).parent / "sanskrit_compounds_benchmark.json"
    
    if not compounds_file.exists():
        logger.warning(
            f"Compounds file not found at {compounds_file}. "
            "GRETIL test cases will not be included."
        )
        return []
        
    try:
        suite = GretilTestSuite(str(compounds_file))  # Convert Path to string for compatibility
        return suite.generate_test_cases()
    except Exception as e:
        logger.error(f"Error loading GRETIL test cases: {str(e)}")
        return []