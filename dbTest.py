from functions.SQLiteFind import SQLite_find_name, SQLite_find_verb, optimized_find_name, optimized_find_verb

# Helper function to measure query performance
def measure_query_performance(func, word, iterations=100):
    """
    Measure the average execution time of a query function.
    
    Args:
        func: The function to test
        word: The word to look up
        iterations: Number of times to run the test
        
    Returns:
        float: Average execution time in milliseconds
    """
    import time
    
    total_time = 0
    for _ in range(iterations):
        start_time = time.time()
        func(word)
        total_time += (time.time() - start_time)
    
    return (total_time / iterations) * 1000  # Convert to milliseconds


# Test performance of original vs optimized functions
test_word = "pratiprasave"  # Or any other word you want to test with

old_time = measure_query_performance(SQLite_find_name, test_word)
new_time = measure_query_performance(optimized_find_name, test_word)

print(f"Original function average time: {old_time:.2f}ms")
print(f"Optimized function average time: {new_time:.2f}ms")
print(f"Performance improvement: {((old_time - new_time) / old_time * 100):.2f}%")


test_word = "gacchathaḥ"  # Or any other word you want to test with

old_time = measure_query_performance(SQLite_find_verb, test_word)
new_time = measure_query_performance(optimized_find_verb, test_word)

print(f"Original function average time: {old_time:.2f}ms")
print(f"Optimized function average time: {new_time:.2f}ms")
print(f"Performance improvement: {((old_time - new_time) / old_time * 100):.2f}%")