import setuptools

# --- Configuration ---
# Package specific info
PACKAGE_NAME = "process_sanskrit" # The actual package name
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
    packages=setuptools.find_packages(), # Find packages in the current directory
    # Define required dependencies
    install_requires=[
        "requests>=2.20", # Need requests for downloading
        "sqlalchemy>=1.4.0", # For database operations
        "pandas>=1.0.0", # For data processing
        "indic_transliteration>=2.0.0", # For transliteration
        "regex>=2022.0.0", # For regex operations
        "python-dotenv>=0.19.0", # For environment variables
        "sanskrit_parser>=0.1.1",
        "gensim>=4.0.0",
        "sentencepiece>=0.1.95",
    ],
    # Define optional dependencies
    extras_require={
        'api': [
            "flask>=2.0.0",
            "flask_cors>=3.0.0",
            "openai>=0.27.0",
        ],
        'byt5': [
            "torch>=1.9.0",
            "transformers>=4.5.0",
        ],
        'sandhi': [
            "sanskrit_parser>=0.1.1",
        ],
        'scoring': [
            "gensim>=4.0.0",
            "sentencepiece>=0.1.95",
        ],
        'all': [
            "flask>=2.0.0",
            "flask_cors>=3.0.0",
            "openai>=0.27.0",
            "torch>=1.9.0",
            "transformers>=4.5.0",
            "sanskrit_parser>=0.1.1",
            "gensim>=4.0.0",
            "sentencepiece>=0.1.95",
        ],
    },
    # Define the command-line script for updates
    entry_points={
        'console_scripts': [
            # The command name users type, points to the function in the setup module
            f'update-ps-database={PACKAGE_NAME}.setup.updateDB:update_database',
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