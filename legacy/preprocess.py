def preprocess(text):

    remove_char_text = ''.join(c for c in text if c.isalpha() or c == "'" or c == ' ')

    print("processing", text)
    ## attempt to remove sandhi and tokenise in any case
    splitted_text = sandhi_splitter(remove_char_text)   

    splitted_text = remove_stopwords_list(splitted_text)    
    
    inflections = inflect(splitted_text) 
    
    entry_list = []
    entry_list = [entry[0] for entry in inflections if len(entry[0]) > 1]
    entry_list = [key for key, group in groupby(entry_list)]

    i = 0
    while i < len(entry_list) - 1:
        # Normalize words to form without diacritics
        word1 = unicodedata.normalize('NFD', entry_list[i])
        word2 = unicodedata.normalize('NFD', entry_list[i+1])

        # Remove diacritics
        word1 = ''.join(c for c in word1 if not unicodedata.combining(c))
        word2 = ''.join(c for c in word2 if not unicodedata.combining(c))

        if word1 in word2 or word2 in word1:
            if len(entry_list[i]) > len(entry_list[i+1]):
                entry_list.pop(i)
            else:
                entry_list.pop(i+1)
        else:
            i += 1
    print("processed", entry_list)
    return clean_results(entry_list)