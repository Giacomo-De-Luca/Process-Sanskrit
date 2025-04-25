import setuptools
import os

# --- Configuration ---
# Package specific info
PACKAGE_NAME = "process_sanskrit" # The actual package name (directory in src/)
# --- End Configuration ---

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A package for processing Sanskrit, including dictionaries."

setuptools.setup(
    name="process-sanskrit",  # How users pip install it (pip install process-sanskrit)
    version="1.0.0",          # Corresponds to your release tag? Keep in sync.
    author="Giacomo De Luca", # Your name
    author_email="your_email@example.com", # Your email
    description="A package for processing Sanskrit, including dictionaries.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Giacomo-De-Luca/Process-Sanskrit", # Link to repo
    package_dir={"": "src"}, # Tell setuptools packages are under src/
    packages=setuptools.find_packages(where="src"), # Find packages in src/
    # Define required dependencies
    install_requires=[
        "requests>=2.20", # Need requests for downloading
        # Add other dependencies your package needs
        # e.g., "regex", "pandas", etc.
    ],
    # Define the command-line script for updates
    entry_points={
        'console_scripts': [
            # The command name users type, points to the function in the new file
            f'update-process-sanskrit-db={PACKAGE_NAME}.download_cli:update_database_command',
        ],
    },
    # No custom cmdclass needed anymore
    # No package_data needed for the large DB as it's downloaded separately

    python_requires='>=3.8', # importlib.resources.files requires Python 3.8+
    classifiers=[ # Trove classifiers - helps users find your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta", # Or appropriate status
        "Topic :: Text Processing :: Linguistic",
    ],
)