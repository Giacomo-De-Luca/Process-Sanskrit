import json

import importlib.resources
import os

# Use importlib.resources to get the correct path to the resources
def get_resource_path(resource_name):
    """Get the path to a resource file using importlib.resources"""
    try:
        # For Python 3.9+
        return str(
            importlib.resources.files('process_sanskrit.resources').joinpath(
                resource_name
            )
        )
    except (AttributeError, ImportError):
        # Fallback for older Python versions
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_dir, 'resources', resource_name)

_mwdictionary_keys = None


def _load_mw_dictionary_keys():
    """Load the optional MW key list only when a caller explicitly requests it."""
    global _mwdictionary_keys
    if _mwdictionary_keys is None:
        with open(
            get_resource_path('MWKeysOnly.json'), 'r', encoding='utf-8'
        ) as resource_file:
            _mwdictionary_keys = json.load(resource_file)
    return _mwdictionary_keys


def __getattr__(name):
    # Preserve ``from ...loadResources import mwdictionaryKeys`` without making
    # every morphology lookup pay the cost of parsing the 4.1 MB JSON file.
    if name == "mwdictionaryKeys":
        return _load_mw_dictionary_keys()
    raise AttributeError(name)


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

