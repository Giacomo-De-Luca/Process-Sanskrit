import re

# Load the text file
with open('Books/raw_books/vivarana_translit.txt', 'r') as f:
    text = f.read()

# Find all lines that contain the pattern '||(anything here)|'
matches = re.findall(r'^(.*\|\|.*?\|.*)$', text, re.MULTILINE)

# Print the matches
for match in matches:
    print(match)