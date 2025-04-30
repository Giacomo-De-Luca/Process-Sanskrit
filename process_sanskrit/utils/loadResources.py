import json 

import importlib.resources
import os
from pathlib import Path

# Use importlib.resources to get the correct path to the resources
def get_resource_path(resource_name):
    """Get the path to a resource file using importlib.resources"""
    try:
        # For Python 3.9+
        with importlib.resources.files('process_sanskrit.resources').joinpath(resource_name) as path:
            return str(path)
    except (AttributeError, ImportError):
        # Fallback for older Python versions
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_dir, 'resources', resource_name)

# Load the dictionary keys
with open(get_resource_path('MWKeysOnly.json'), 'r', encoding='utf-8') as f:
    mwdictionaryKeys = json.load(f)


def load_type_map(file_path):
    """Load the type mapping from TSV file into a dictionary."""
    type_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split by tab and extract the first two columns
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                abbr, description = parts[0], parts[1]
                type_map[abbr] = description
    return type_map

type_map = load_type_map(get_resource_path('type_map.tsv'))


