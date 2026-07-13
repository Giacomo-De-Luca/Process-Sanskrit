from process_sanskrit.functions.process import process
from process_sanskrit.functions.dictionaryLookup import dict_search
from process_sanskrit.utils.transliterationUtils import transliterate


import logging
import warnings
import io
import sys
import os # Needed for os.devnull

from contextlib import contextmanager, redirect_stdout, redirect_stderr


# 1. Configure the root logger to only show CRITICAL messages
logging.getLogger().setLevel(logging.CRITICAL)
# 2. Permanently silence the vendored sandhi splitter, which logs every sandhi
#    rule it tries at DEBUG. (Formerly 'sanskrit_parser'; see splitter/NOTICE.md.)
sp_logger = logging.getLogger('process_sanskrit.splitter')
sp_logger.addHandler(logging.NullHandler()) # Add handler that does nothing
sp_logger.propagate = False # Prevent messages going to the root logger
sp_logger.setLevel(logging.CRITICAL)

# 3. Disable warnings from SQLAlchemy
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