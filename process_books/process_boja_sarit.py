import xml.etree.ElementTree as ET
import re

def get_all_text(element):
    text = element.text or ''
    for child in element:
        if child.tag == '{http://www.tei-c.org/ns/1.0}note':
            if child.tail:
                text += child.tail
            continue
        text += get_all_text(child)
        if child.tail:
            text += child.tail
    return text

# Parse the XML file
tree = ET.parse('/Users/jack/Desktop/boja.xml')
root = tree.getroot()

# Define the namespace
namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

# Find all 'div' elements
divs = root.findall(".//tei:div", namespaces)

# Initialize the last sutra div and the commentary count
last_sutra_div = None
commentary_count = 0

# Initialize the dictionary
sutra_dict = {}

# For each 'div' element
for div in divs:
    type = div.get('type')
    if type == 'sūtra' or type == 'sutra':
        # If it's a sutra div, get its id and text, and reset the commentary count
        sutra_id = div.get('{http://www.w3.org/XML/1998/namespace}id')
        sutra_key = re.search(r'\.(\d+\.\d+)$', sutra_id)  # Get the numbers before and after the last dot
        if sutra_key:
            sutra_key = sutra_key.group(1)
        else:
            continue
        p = div.find('tei:p', namespaces)
        if p is not None:
            sutra_text = get_all_text(p)
        else:
            sutra_text = ''
        last_sutra_div = div
        commentary_count = 0
        sutra_dict[sutra_key] = {'sutra_text': sutra_text, 'commentary_text': []}
    elif type == 'commentary' and last_sutra_div is not None and commentary_count < 2:
        # If it's a commentary div and there was a sutra div before it and we haven't printed two commentaries yet,
        # get its text, and increment the commentary count
        p = div.find('tei:p', namespaces)
        if p is not None:
            commentary_text = get_all_text(p)
        else:
            commentary_text = ''
        commentary_count += 1
        sutra_dict[sutra_key]['commentary_text'].append(commentary_text)

# Clean the dictionary
for key in sutra_dict:
    sutra_dict[key]['sutra_text'] = re.sub(r'\|\|.*?\|\|', '', sutra_dict[key]['sutra_text'])
    sutra_dict[key]['commentary_text'] = [re.sub(r'\|\|.*?\|\|', '', text) for text in sutra_dict[key]['commentary_text']]
    sutra_dict[key]['commentary_text'] = [text.replace('vṛttiḥ ---', '', 1).rstrip() for text in sutra_dict[key]['commentary_text']]
    sutra_dict[key]['commentary_text'] = [re.sub(r'\s+', ' ', text) for text in sutra_dict[key]['commentary_text']]
    sutra_dict[key]['commentary_text'] = [text.replace('|', '|\n') for text in sutra_dict[key]['commentary_text']]
    sutra_dict[key]['commentary_text'] = "\n\n".join(sutra_dict[key]['commentary_text'])

print(sutra_dict["1.1"]["commentary_text"])

import json

# Save the dictionary as a JSON file
with open('boja_commentary.json', 'w') as f:
    json.dump(sutra_dict, f, indent=4)