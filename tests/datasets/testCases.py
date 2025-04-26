## my first attempt to create a test dataset with manually picked examples
## to small to be actually relevant, but it can be useful for debugging

test_cases = [
            # Basic Dvandva (Co-ordinative) Compounds
            {
                "input": "rāmalakṣmaṇau",
                "correct_split": ["rāma", "lakṣmaṇa"],
                "type": "dvandva",
                "complexity": "simple"
            },
            {
                "input": "devamanuṣyāḥ",
                "correct_split": ["deva", "manuṣya"],
                "type": "dvandva",
                "complexity": "simple"
            },
            
            # Simple Tatpuruṣa (Determinative) Compounds
            {
                "input": "rājapuruṣaḥ",
                "correct_split": ["rāja", "puruṣa"],
                "type": "tatpurusha",
                "complexity": "simple"
            },
            {
                "input": "dharmajñānam",
                "correct_split": ["dharma", "jñāna"],
                "type": "tatpurusha",
                "complexity": "simple"
            },

            # Compounds with Indeclinables
            {
                "input": "yathāśakti",
                "correct_split": ["yathā", "śakti"],
                "type": "avyayibhava",
                "complexity": "simple"
            },
            {
                "input": "yadṛcchayā",
                "correct_split": ["yad", "ṛcchayā"],
                "type": "with_indeclinable",
                "complexity": "simple"
            },

            # Compounds with Prefixes
            {
                "input": "pratyakṣapramāṇam",
                "correct_split": ["prati", "akṣa", "pramāṇa"],
                "type": "with_prefix",
                "complexity": "medium"
            },
            {
                "input": "anupapannam",
                "correct_split": ["an", "upapanna"],
                "type": "with_prefix",
                "complexity": "simple"
            },

            # Medium Length Compounds
            {
                "input": "dharmārthakāmamokṣāḥ",
                "correct_split": ["dharma", "artha", "kāma", "mokṣā"],
                "type": "dvandva",
                "complexity": "medium"
            },
            {
                "input": "śabdajñānānupātī",
                "correct_split": ["śabda", "jñāna", "anupāti"],
                "type": "tatpurusha",
                "complexity": "medium"
            },

            # Complex Philosophical Compounds
            {
                "input": "svarūpaśūnyevārthamātranirbhāsā",
                "correct_split": ["svarūpa", "śūnya", "iva", "arthamātra", "ni", "bhās"],
                "type": "complex_with_indeclinable",
                "complexity": "high"
            },
            {
                "input": "cittavṛttinirodhaḥ",
                "correct_split": ["citta", "vṛtti", "nirodha"],
                "type": "philosophical",
                "complexity": "medium"
            },

            # Long Technical Compounds
            {
                "input": "dharmalakṣaṇāvasthāpariṇāmā",
                "correct_split": ["dharma", "lakṣaṇa", "avasthā", "pariṇāmā"],
                "type": "technical",
                "complexity": "high"
            },
            {
                "input": "sattvaśuddhisaumanasyaikāgryendriyajayātmadarśanayogyatvāni",
                "correct_split": ["sattva", "śuddhi", "saumanasya", "ekāgrya", "indriya", "jaya", "ātma", "darśana", "yogyatva"],
                "type": "technical",
                "complexity": "very_high"
            },

            # Compounds with Tricky Sandhis
            {
                "input": "tajjñānam",
                "correct_split": ["tad", "jñāna"],
                "type": "consonant_sandhi",
                "complexity": "simple"
            },
            {
                "input": "tacchabdaḥ",
                "correct_split": ["tad", "śabda"],
                "type": "consonant_sandhi",
                "complexity": "simple"
            },

            # Cases with Multiple Valid Splits
            {
                "input": "rājapuruṣottamaḥ",
                "correct_split": ["rāja", "puruṣa", "uttama"],  # Could also be ["rāja", "puruṣottamaḥ"]
                "type": "ambiguous",
                "complexity": "medium"
            }
        ]