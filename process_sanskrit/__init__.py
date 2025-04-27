from process_sanskrit.functions.process import process
from process_sanskrit.functions.dictionaryLookup import get_voc_entry
from process_sanskrit.utils.transliterationUtils import transliterate


import logging
import warnings
import io
import sys
import os # Needed for os.devnull

from contextlib import contextmanager, redirect_stdout, redirect_stderr


# 1. Configure the root logger to only show CRITICAL messages
logging.getLogger().setLevel(logging.CRITICAL)
# 2. Permanently silence 'sanskrit_parser' and its children
sp_logger = logging.getLogger('sanskrit_parser')
sp_logger.addHandler(logging.NullHandler()) # Add handler that does nothing
sp_logger.propagate = False # Prevent messages going to the root logger
# Setting level is now less critical, but doesn't hurt
sp_logger.setLevel(logging.CRITICAL)
# The loop for submodules is also less critical if propagation is false, but kept for clarity
for submodule in ['parser', 'util', 'lexical_analyzer', 'base', 'sandhi_analyzer']:
    sub_logger = logging.getLogger(f'sanskrit_parser.{submodule}')
    sub_logger.addHandler(logging.NullHandler())
    sub_logger.propagate = False
    sub_logger.setLevel(logging.CRITICAL)


# 3. Permanently silence 'sanskrit_util'
su_logger = logging.getLogger('sanskrit_util')
su_logger.addHandler(logging.NullHandler()) # Add handler that does nothing
su_logger.propagate = False # Prevent messages going to the root logger
su_logger.setLevel(logging.CRITICAL) # Set level (less critical now)

# 4. Disable warnings from SQLAlchemy
warnings.filterwarnings('ignore', category=UserWarning, module='sqlalchemy')

# 5. Silence all other warnings
warnings.filterwarnings('ignore')


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