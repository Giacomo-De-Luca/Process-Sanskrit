
import json 
import pandas as pd

with open('resources/MWKeysOnly.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeys = json.load(f)

with open('resources/MWKeysOnlySLP1.json', 'r', encoding='utf-8') as f:
    mwdictionaryKeysSLP1 = json.load(f)

## also map to the type.
# Read the Excel file into a DataFrame
type_map = pd.read_excel('resources/type_map.xlsx')

