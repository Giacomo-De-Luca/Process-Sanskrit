

def root_any_list(text_list):
    roots = []
    for word in text_list:
        root_found = root_any_word(word)
        if root_found is not None:
            roots.append(root_found)
    
    # Transliterate the first element of each tuple
    for i in range(len(roots)):
        roots[i] = (transliterateSLP1IAST(roots[i][0].replace('-', '')),) + roots[i][1:]
    
    return roots
