"""Compare ways of reading compound members for lemma search.

Settings live in benchmarks/compound-members.json; the DCS directory comes from
PROCESS_SANSKRIT_DCS_CONLLU_DIR.  Run from the repository root:

    PROCESS_SANSKRIT_DCS_CONLLU_DIR=/path/to/dcs/conllu/files \\
        uv run python scripts/evaluate_compound_members.py

See scripts/compound_members/README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compound_members.evaluation import CompoundMemberEvaluation, EvaluationConfiguration  # noqa: E402


if __name__ == "__main__":
    CompoundMemberEvaluation(EvaluationConfiguration.load()).run()
