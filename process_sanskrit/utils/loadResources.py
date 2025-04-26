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

## let's get rid of the xlsx file
## so I can avoid using pandas 
## also a dictionary makes for faster lookups
# 
import pandas as pd
# Read the Excel file into a DataFrame
type_map = pd.read_excel(get_resource_path('type_map.xlsx'))
