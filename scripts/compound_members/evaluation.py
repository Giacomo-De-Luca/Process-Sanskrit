"""Runs the compound-member evaluation that benchmarks/compound-members.json describes.

`CompoundMemberEvaluation.run` does, in order:

1. read the dictionaries once (`Resources`) and build every variant;
2. compute every variant's members for every headword with components, and
   note which headwords each variant changes against the baseline;
3. read DCS once: evidence for every headword some variant gives members,
   and the hand-checked cases, plus a sample of running text;
4. analyse with the library what the scorers need (`LibraryAnalyses`);
5. score every variant and write the report.
"""

import json
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from process_sanskrit.utils.resourcePaths import get_database_path

from process_sanskrit.functions.lemmaAnalysis import PieceLemmas
from process_sanskrit.utils.compoundMembers import CompoundForms
from process_sanskrit.utils.databaseSetup import get_session
from process_sanskrit.utils.lexicalResources import pausa_readings

from .analyses import LibraryAnalyses
from .dcs import ConlluReader, DcsEvidence
from .joiner import SandhiJoiner
from .lexicon import ComponentTables, Headwords, read_only_connection
from .readings import MemberVariant, VariantFactory
from .scoring import CaseBook, MemberScorer, RetrievalSimulation, RunningTextImpact

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvaluationConfiguration:
    root: Path
    dcs_directory: Optional[Path]
    cases_path: Path
    sample: Mapping
    analysis: Mapping
    output: Mapping
    filter_profiles: Mapping
    variants: Sequence[Mapping]

    DEFAULT = ROOT / "benchmarks" / "compound-members.json"

    @classmethod
    def load(cls, path: Path = DEFAULT) -> "EvaluationConfiguration":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported schema_version {raw.get('schema_version')!r}")
        if not raw["variants"] or raw["variants"][0]["name"] != "baseline":
            raise ValueError(f"{path}: the first variant must be the baseline")
        return cls(
            ROOT, cls._directory(raw["corpora"]["dcs_conllu_directory"]), ROOT / raw["cases"],
            raw["sample"], raw["analysis"], raw["output"], raw["filter_profiles"], raw["variants"],
        )

    @staticmethod
    def _directory(value: str) -> Optional[Path]:
        expanded = os.path.expandvars(value)
        return None if not expanded or "$" in expanded else Path(expanded).expanduser()

    def path(self, relative: str) -> Path:
        return self.root / relative


class Resources:
    """The dictionaries, read once and shared by every variant and scorer.

    The library's own objects where the library has them: `CompoundForms`
    and `PieceLemmas`, the lemmas lemma mode gives a piece on its own (the
    guard of the citation forms).  `joiner` joins DCS segments; it loads the
    vendored splitter's rules (about 200 MB), so only on first use.
    """

    def __init__(self):
        connection = read_only_connection(get_database_path())
        self.tables = ComponentTables(connection)
        connection.close()
        self.session = get_session()
        self.forms = CompoundForms(self.session)
        self.piece_lemmas = PieceLemmas(self.session)
        self.headwords = Headwords()

    @property
    def joiner(self) -> SandhiJoiner:
        return SandhiJoiner.shared()

    def factory(self, config: EvaluationConfiguration) -> VariantFactory:
        return VariantFactory(self.tables, self.headwords, self.forms, config.filter_profiles, self.piece_lemmas)


class CompoundMemberEvaluation:
    def __init__(self, config: EvaluationConfiguration):
        self.config = config
        self.started = time.time()

    @staticmethod
    def variant_factory(config: EvaluationConfiguration) -> VariantFactory:
        """A factory over freshly read dictionaries (for tests and one-off checks)."""
        return Resources().factory(config)

    def log(self, message: str) -> None:
        print(f"[{time.time() - self.started:7.1f}s] {message}", file=sys.stderr, flush=True)

    def run(self) -> dict:
        config, sample = self.config, self.config.sample
        if config.dcs_directory is None or not config.dcs_directory.is_dir():
            raise SystemExit("Set PROCESS_SANSKRIT_DCS_CONLLU_DIR to the DCS CoNLL-U directory (one folder per work).")
        rng = random.Random(sample["seed"])

        self.log("reading the dictionaries")
        resources = Resources()
        analyses = LibraryAnalyses(config.path(config.analysis["store"]), config.analysis["workers"])
        if analyses.discarded:
            self.log("the analysis store was made by another library, algorithm or database: discarded")
        factory = resources.factory(config)
        variants = [factory.build(spec) for spec in config.variants]
        baseline = variants[0]
        cases = CaseBook.load(config.cases_path)
        case_headwords = {case.headword for case in cases.cases}

        universe = sorted(resources.tables.headwords() | case_headwords)
        self.log(f"members of {len(universe)} headwords under {len(variants)} variants")
        changed = {variant.name: [h for h in universe if variant.members(h) != baseline.members(h)]
                   for variant in variants[1:]}
        affected = set().union(*changed.values())
        wanted = {h for h in universe if any(variant.members(h) for variant in variants)} | case_headwords

        self.log(f"reading DCS for {len(wanted)} headwords")
        evidence = DcsEvidence(resources.joiner, wanted, sample["max_run"])
        running: Counter = Counter()
        sentences = tokens = 0
        for sentence in ConlluReader(config.dcs_directory).sentences():
            sentences += 1
            tokens += len(sentence.tokens)
            evidence.add(sentence)
            if zlib.crc32(f"{sample['seed']}:{sentence.work}:{sentence.sent_id}".encode()) % sample["running_text_one_in"] == 0:
                running.update(token.surface for token in sentence.tokens)
            if sentences % 100_000 == 0:
                self.log(f"  {sentences} sentences")
        self.log(f"DCS: {sentences} sentences, {tokens} tokens; evidence for {len(evidence)} headwords")

        self.log(f"analysing {len(running)} running-text tokens")
        analyses.analyse(running)
        judged = [h for h in sorted(wanted) if evidence.get(h) and evidence.get(h).occurrences]
        pieces = {reading for headword in judged
                  for _, forms in evidence.get(headword).gold_splits(sample["gold_splits"])
                  for form in forms for reading in pausa_readings(form)}
        self.log(f"analysing {len(pieces)} spellings of gold segments")
        analyses.analyse(pieces)
        retrieval, weights, strata, description = self.retrieval_sample(changed, wanted, affected, case_headwords, evidence, rng)
        simulation = RetrievalSimulation(analyses, sample["lines_per_headword"], sample["query_forms_per_headword"])
        words = simulation.words(retrieval)
        self.log(f"analysing {len(words)} words for the retrieval simulation ({len(retrieval)} headwords)")
        analyses.analyse(words)

        self.log("scoring")
        scorers = {yardstick: MemberScorer(resources.joiner, analyses.lemma_set, sample["gold_splits"], yardstick)
                   for yardstick in MemberScorer.YARDSTICKS}
        impact = RunningTextImpact(analyses, scorers["index"], evidence)
        baseline_reached = simulation.reached(baseline.members, retrieval)
        report = {
            "meta": {
                "dcs_sentences": sentences, "dcs_tokens": tokens,
                "running_text_tokens": sum(running.values()),
                "universe": len(universe), "compound_form_stubs": len(resources.forms.stubs()),
                "affected": len(affected), "wanted": len(wanted),
                "judged_in_dcs": len(judged), "analysis_errors": len(analyses.errors),
                "retrieval_headwords": len(retrieval), "retrieval_strata": description,
            },
            "variants": {},
        }
        for variant in variants:
            report["variants"][variant.name] = self.score_variant(
                variant, baseline, changed.get(variant.name, []), judged, evidence, scorers,
                simulation, retrieval, weights, strata, baseline_reached, impact, running, cases,
            )
            self.log(f"  scored {variant.name}")
        report["meta"]["seconds"] = round(time.time() - self.started, 1)
        self.write(report, evidence)
        return report

    def retrieval_sample(self, changed: Mapping[str, List[str]], wanted: Set[str], affected: Set[str],
                         case_headwords: Set[str], evidence: DcsEvidence, rng: random.Random):
        """Headwords for the simulation, stratified by which variants change them.

        A headword's stratum is the set of variants that change its members;
        headwords that have members and no variant changes form the stratum
        `unchanged`.  Each stratum is sampled up to `retrieval_per_stratum`,
        and a headword weighs as many headwords as its stratum has per draw.
        Strata smaller than `retrieval_strata_minimum` are pooled.  The
        hand-checked cases are always included, with weight 0 unless drawn.
        """
        per_stratum = self.config.sample["retrieval_per_stratum"]
        minimum = self.config.sample["retrieval_strata_minimum"]
        signature: Dict[str, Tuple[str, ...]] = defaultdict(tuple)
        for name, headwords in changed.items():
            for headword in headwords:
                signature[headword] += (name,)
        for headword in wanted - affected - case_headwords:
            signature[headword] = ("unchanged",)
        grouped: Dict[str, List[str]] = defaultdict(list)
        for headword, names in signature.items():
            found = evidence.get(headword)
            if found and found.occurrences:
                grouped[" + ".join(names)].append(headword)
        pooled: Dict[str, List[str]] = defaultdict(list)
        for name, headwords in grouped.items():
            pooled[name if len(headwords) >= minimum or name == "unchanged" else "other"].extend(headwords)
        sample, weights, strata, description = {}, {}, {}, {}
        for name in sorted(pooled):
            headwords = sorted(pooled[name])
            drawn = rng.sample(headwords, min(per_stratum, len(headwords)))
            for headword in drawn:
                sample[headword] = evidence.get(headword)
                weights[headword] = len(headwords) / len(drawn)
                strata[headword] = name
            description[name] = {"headwords": len(headwords), "drawn": len(drawn)}
        for headword in sorted(case_headwords):
            if evidence.get(headword) and headword not in sample:
                sample[headword] = evidence.get(headword)
                weights[headword] = 0.0
                strata[headword] = "cases"
        return sample, weights, strata, description

    def score_variant(self, variant: MemberVariant, baseline: MemberVariant, changed: Sequence[str],
                      judged: Sequence[str], evidence: DcsEvidence, scorers: Mapping[str, MemberScorer],
                      simulation: RetrievalSimulation, retrieval, weights, strata, baseline_reached,
                      impact: RunningTextImpact, running: Mapping[str, int], cases: CaseBook) -> dict:
        judgments = {}
        for yardstick, scorer in scorers.items():
            counts = {"headwords": Counter(), "occurrences": Counter(), "split_occurrences": Counter()}
            for headword in judged:
                found = evidence.get(headword)
                judgment = scorer.judge(variant.members(headword), found)
                counts["headwords"][judgment] += 1
                counts["occurrences"][judgment] += found.occurrences
                counts["split_occurrences"][judgment] += sum(found.splits.values())
            judgments[yardstick] = {
                weighting: dict(counter) for weighting, counter in counts.items()
            }
            judgments[yardstick]["errors"] = {
                weighting: sum(counts[weighting][name] for name in MemberScorer.ERRORS) for weighting in counts
            }
        scorer = scorers["index"]
        transitions = {"headwords": Counter(), "occurrences": Counter()}
        examples: Dict[str, List[dict]] = defaultdict(list)
        limit = self.config.sample["examples_per_change"]
        ## examples are a seeded random sample of each change, not its alphabetical start
        shuffled = list(changed)
        random.Random(f"{self.config.sample['seed']}:{variant.name}").shuffle(shuffled)
        for headword in shuffled:
            found = evidence.get(headword)
            key = f"{scorer.judge(baseline.members(headword), found)}->{scorer.judge(variant.members(headword), found)}"
            transitions["headwords"][key] += 1
            transitions["occurrences"][key] += found.occurrences if found else 0
            if len(examples[key]) < limit:
                examples[key].append({
                    "headword": headword, "baseline": baseline.members(headword),
                    "variant": variant.members(headword), "dcs": found.as_dict(surfaces=3) if found else None,
                })
        reached = simulation.reached(variant.members, retrieval)
        return {
            "description": variant.description,
            "changed_headwords": len(changed),
            "changed_in_dcs": sum(1 for h in changed if evidence.get(h) and evidence.get(h).occurrences),
            "judgments": judgments,
            "transitions": {weighting: dict(counter) for weighting, counter in transitions.items()},
            "retrieval": simulation.compare(reached, baseline_reached, weights, strata,
                                            self.config.sample["bootstrap_resamples"], self.config.sample["seed"]),
            "retrieval_reached": reached,
            "running_text": impact.score(variant.members, baseline.members, running),
            "cases": cases.score(variant.members),
            "examples": dict(examples),
        }

    def write(self, report: dict, evidence: DcsEvidence) -> None:
        output = self.config.output
        report_path = self.config.path(output["report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        evidence_path = self.config.path(output["evidence"])
        evidence_path.write_text(json.dumps(
            {headword: found.as_dict() for headword, found in sorted(evidence.items())}, ensure_ascii=False
        ), encoding="utf-8")
        summary_path = self.config.path(output["summary"])
        summary_path.write_text(EvaluationSummary(report).markdown(), encoding="utf-8")
        self.log(f"wrote {report_path}, {summary_path}")


class EvaluationSummary:
    """The report's headline numbers as Markdown tables."""

    def __init__(self, report: dict):
        self.report = report

    @staticmethod
    def _added(result: dict) -> int:
        """DCS occurrences of words DCS never splits that the variant newly gives members."""
        return sum(weight for key, weight in result["transitions"]["occurrences"].items()
                   if key.endswith("->unverified") and not key.startswith("unverified"))

    def markdown(self) -> str:
        meta = self.report["meta"]
        variants = self.report["variants"]
        base = variants["baseline"]["judgments"]
        lines = [
            "# Compound members: evaluation summary", "",
            f"DCS: {meta['dcs_sentences']:,} sentences, {meta['dcs_tokens']:,} tokens. "
            f"Headwords with components: {meta['universe']:,}; changed by some variant: {meta['affected']:,}; "
            f"judged against DCS: {meta['judged_in_dcs']:,}.", "",
            "## Errors against DCS (wrong + missing members)", "",
            "Weighted by all DCS occurrences of the headword, and by its split occurrences only. "
            "`index`: members the index holds for a split compound; `lemma`: citation lemmas.", "",
            "| variant | changed | index, all | Δ | index, split | Δ | lemma, all | Δ | unverified added |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, result in variants.items():
            index, lemma = result["judgments"]["index"]["errors"], result["judgments"]["lemma"]["errors"]
            lines.append(
                f"| {name} | {result['changed_headwords']:,} | {index['occurrences']:,} | "
                f"{index['occurrences'] - base['index']['errors']['occurrences']:+,} | {index['split_occurrences']:,} | "
                f"{index['split_occurrences'] - base['index']['errors']['split_occurrences']:+,} | "
                f"{lemma['occurrences']:,} | {lemma['occurrences'] - base['lemma']['errors']['occurrences']:+,} | "
                f"{self._added(result):,} |"
            )
        lines += ["", "## Search simulation against the baseline (DCS occurrences reached)", "",
                  "Gained/lost in the sample; net weighted to the strata, with a 95% bootstrap interval.", "",
                  "| variant | stem: gained / lost | stem: net [95%] | form: gained / lost | form: net [95%] |",
                  "|---|---:|---:|---:|---:|"]
        for name, result in variants.items():
            stem, form = result["retrieval"]["stem"], result["retrieval"]["form"]
            lines.append(
                f"| {name} | +{stem['gained_in_sample']} / −{stem['lost_in_sample']} | "
                f"{stem['net_weighted']:+,} [{stem['interval'][0]:+,}, {stem['interval'][1]:+,}] | "
                f"+{form['gained_in_sample']} / −{form['lost_in_sample']} | "
                f"{form['net_weighted']:+,} [{form['interval'][0]:+,}, {form['interval'][1]:+,}] |"
            )
        lines += ["", "## Hand-checked cases", "", "| variant | correct | acceptable | missed | spurious | wrong |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, result in variants.items():
            totals = result["cases"]["totals"]
            lines.append(f"| {name} | {totals['correct']} | {totals['acceptable']} | {totals['missed']} | "
                         f"{totals['spurious']} | {totals['wrong']} |")
        return "\n".join(lines) + "\n"
