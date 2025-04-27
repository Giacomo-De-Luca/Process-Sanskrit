from process_sanskrit.functions.process import process
from process_sanskrit.functions.dictionaryLookup import get_voc_entry
from process_sanskrit.utils.transliterationUtils import transliterate


import logging
import warnings
import io
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# 1. Configure the root logger to only show CRITICAL messages
logging.getLogger().setLevel(logging.CRITICAL)

# 2. Specifically configure the sanskrit_parser loggers at all levels
logging.getLogger('sanskrit_parser').setLevel(logging.CRITICAL)
for submodule in ['parser', 'util', 'lexical_analyzer', 'base', 'sandhi_analyzer']:
    logging.getLogger(f'sanskrit_parser.{submodule}').setLevel(logging.CRITICAL)

# 3. Configure the sanskrit_util logger (for the SAWarning message)
logging.getLogger('sanskrit_util').setLevel(logging.CRITICAL)

# 4. Disable warnings from SQLAlchemy
warnings.filterwarnings('ignore', category=UserWarning, module='sqlalchemy')

# 5. Silence all other warnings
warnings.filterwarnings('ignore')

# 6. Filter out gensim/sentencepiece warning specifically
warnings.filterwarnings('ignore', message='gensim and/or sentencepiece not found')

# 7. Create a context manager to suppress all output temporarily
@contextmanager
def suppress_all_output():
    """
    A context manager that redirects stdout and stderr to devnull,
    effectively silencing all console output.
    """
    with open(os.devnull, 'w') as null:
        with redirect_stdout(null), redirect_stderr(null):
            yield

# 8. Optional: Disable SQLAlchemy logging more aggressively
logging.getLogger('sqlalchemy').setLevel(logging.CRITICAL)