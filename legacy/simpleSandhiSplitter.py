
def sandhi_splitter(text_to_split, cached=True, attempts=1):
    """
    Splits the given text using a sandhi splitter parser.
    Checks if the result is already cached before performing the split.

    Parameters:
    - text_to_split (str): The text to split.

    Returns:
    - list: A list of split parts of the text.
    """

    #text_to_split = anythingToSLP1(text_to_split).strip()

    # Check if the result is already cached
    if cached == True:
        cached_result = session.query(SplitCache).filter_by(input=text_to_split).first()
    
    if cached == True: 
        if cached_result:
            # Retrieve and return the cached result if it exists
            splitted_text = ast.literal_eval(cached_result.splitted_text)
            print(f"Retrieved from cache: {splitted_text}")
            return splitted_text

    # If not cached, perform the split
    try:
        splits = parser.split(text_to_split, limit=attempts)

        #if split is none, default to split by space
        if splits is None:
            return text_to_split.split()
        
        if attempts == 1: 

            for split in splits:
                splitted_text = f'{split}'
            splitted_text = ast.literal_eval(splitted_text)

        if attempts > 1: 

            splitted_text = []
            for split in splits:
                string_split =  f'{split}'
                splitted_text.append(ast.literal_eval(string_split))



        print(f"Splitted text: {splitted_text}")

        # Store the split result in cache as a list
        if cached == True: 
            new_cache_entry = SplitCache(input=text_to_split, splitted_text=str(splitted_text))
            session.add(new_cache_entry)
            session.commit()
            print(f"Added to cache: {splitted_text}")

        return splitted_text

    except Exception as e:
        print(f"Could not split the line: {text_to_split}")
        print(f"Error: {e}")
        return text_to_split.split()
