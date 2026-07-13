#!/usr/bin/env python3
"""Config-driven, splitter-only correctness and performance benchmark.

The coordinator launches one isolated worker per backend.  Cold samples use a
fresh process each time; the warm corpus sweep reuses one Parser instance.  No
database or downstream morphology code is exercised.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmarks/splitter-benchmark.json"
WORKER_ENV = "PROCESS_SANSKRIT_BENCHMARK_WORKER"
WORKER_BACKEND_ENV = "PROCESS_SANSKRIT_BENCHMARK_BACKEND"
WORKER_CONFIG_ENV = "PROCESS_SANSKRIT_BENCHMARK_CONFIG"


class ConfigurationError(ValueError):
    """Raised when the benchmark configuration is incomplete or ambiguous."""


class JsonDigest:
    """Create stable SHA-256 digests from JSON-compatible values."""

    @staticmethod
    def canonical_bytes(value: Any) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def sha256(cls, value: Any) -> str:
        return hashlib.sha256(cls.canonical_bytes(value)).hexdigest()


class RuntimeProvenance:
    """Compare complete runtime snapshots across isolated workers."""

    @classmethod
    def consistent(cls, runtimes: Sequence[Mapping[str, Any]]) -> bool:
        """Return whether every non-empty worker snapshot is byte-for-byte equal."""
        digests = {JsonDigest.sha256(runtime) for runtime in runtimes}
        return bool(digests) and len(digests) == 1


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    categories: Tuple[str, ...]


@dataclass(frozen=True)
class ExtraCaseSpec:
    text: str
    category: str
    source: str
    warm_repetitions: int


@dataclass(frozen=True)
class LengthBucket:
    name: str
    minimum: int
    maximum: Optional[int]

    def matches(self, length: int) -> bool:
        return length >= self.minimum and (
            self.maximum is None or length <= self.maximum
        )


class BenchmarkConfiguration:
    """Validated benchmark settings loaded from one JSON file."""

    def __init__(
        self,
        *,
        raw: Mapping[str, Any],
        root: Path,
        backends: Tuple[str, ...],
        reference_backend: str,
        backend_environment_variable: str,
        datasets: Tuple[DatasetSpec, ...],
        extra_cases: Tuple[ExtraCaseSpec, ...],
        length_buckets: Tuple[LengthBucket, ...],
        limit: int,
        score: bool,
        input_encoding: str,
        output_encoding: str,
        warm_repetitions: int,
        cold_repetitions: int,
        warmup_cases: Tuple[str, ...],
        cold_case: str,
        require_release_native: bool,
        output_path: Path,
        include_cases: bool,
        fail_on_mismatch: bool,
    ):
        self.raw = raw
        self.root = root
        self.backends = backends
        self.reference_backend = reference_backend
        self.backend_environment_variable = backend_environment_variable
        self.datasets = datasets
        self.extra_cases = extra_cases
        self.length_buckets = length_buckets
        self.limit = limit
        self.score = score
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.warm_repetitions = warm_repetitions
        self.cold_repetitions = cold_repetitions
        self.warmup_cases = warmup_cases
        self.cold_case = cold_case
        self.require_release_native = require_release_native
        self.output_path = output_path
        self.include_cases = include_cases
        self.fail_on_mismatch = fail_on_mismatch
        self.digest = JsonDigest.sha256(raw)

    @classmethod
    def load(cls, path: Path, root: Path = ROOT) -> "BenchmarkConfiguration":
        try:
            mapping = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                "Unable to read benchmark configuration %s: %s" % (path, error)
            ) from error
        return cls.from_mapping(mapping, root)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], root: Path
    ) -> "BenchmarkConfiguration":
        if mapping.get("schema_version") != 1:
            raise ConfigurationError("schema_version must be 1")

        backends = cls._string_tuple(mapping.get("backends"), "backends")
        if len(set(backends)) != len(backends):
            raise ConfigurationError("backends must not contain duplicates")
        reference_backend = cls._string(
            mapping.get("reference_backend"), "reference_backend"
        )
        if reference_backend not in backends:
            raise ConfigurationError("reference_backend must be in backends")
        backend_environment_variable = cls._string(
            mapping.get(
                "backend_environment_variable",
                "PROCESS_SANSKRIT_SPLITTER_BACKEND",
            ),
            "backend_environment_variable",
        )

        dataset_values = mapping.get("datasets")
        if not isinstance(dataset_values, list) or not dataset_values:
            raise ConfigurationError("datasets must be a non-empty list")
        datasets = []
        for index, value in enumerate(dataset_values):
            if not isinstance(value, Mapping):
                raise ConfigurationError("datasets[%d] must be an object" % index)
            relative_path = cls._string(
                value.get("path"), "datasets[%d].path" % index
            )
            datasets.append(
                DatasetSpec(
                    path=cls._resolve(root, relative_path),
                    categories=cls._string_tuple(
                        value.get("categories"),
                        "datasets[%d].categories" % index,
                    ),
                )
            )

        extra_values = mapping.get("extra_cases", [])
        if not isinstance(extra_values, list):
            raise ConfigurationError("extra_cases must be a list")
        extra_cases = []
        for index, value in enumerate(extra_values):
            if not isinstance(value, Mapping):
                raise ConfigurationError("extra_cases[%d] must be an object" % index)
            repetitions = cls._positive_integer(
                value.get("warm_repetitions", 1),
                "extra_cases[%d].warm_repetitions" % index,
            )
            extra_cases.append(
                ExtraCaseSpec(
                    text=cls._string(
                        value.get("text"), "extra_cases[%d].text" % index
                    ),
                    category=cls._string(
                        value.get("category"),
                        "extra_cases[%d].category" % index,
                    ),
                    source=cls._string(
                        value.get("source", "configuration"),
                        "extra_cases[%d].source" % index,
                    ),
                    warm_repetitions=repetitions,
                )
            )

        bucket_values = mapping.get("length_buckets")
        if not isinstance(bucket_values, list) or not bucket_values:
            raise ConfigurationError("length_buckets must be a non-empty list")
        length_buckets = []
        for index, value in enumerate(bucket_values):
            if not isinstance(value, Mapping):
                raise ConfigurationError(
                    "length_buckets[%d] must be an object" % index
                )
            minimum = cls._nonnegative_integer(
                value.get("min_length"),
                "length_buckets[%d].min_length" % index,
            )
            raw_maximum = value.get("max_length")
            maximum = (
                None
                if raw_maximum is None
                else cls._nonnegative_integer(
                    raw_maximum, "length_buckets[%d].max_length" % index
                )
            )
            if maximum is not None and maximum < minimum:
                raise ConfigurationError(
                    "length_buckets[%d] has max_length below min_length" % index
                )
            length_buckets.append(
                LengthBucket(
                    name=cls._string(
                        value.get("name"), "length_buckets[%d].name" % index
                    ),
                    minimum=minimum,
                    maximum=maximum,
                )
            )
        cls._validate_buckets(length_buckets)

        splitter = mapping.get("splitter", {})
        if not isinstance(splitter, Mapping):
            raise ConfigurationError("splitter must be an object")
        limit = cls._nonnegative_integer(splitter.get("limit", 10), "splitter.limit")
        score = splitter.get("score", True)
        if not isinstance(score, bool):
            raise ConfigurationError("splitter.score must be a boolean")
        input_encoding = cls._string(
            splitter.get("input_encoding", "iast"), "splitter.input_encoding"
        )
        output_encoding = cls._string(
            splitter.get("output_encoding", "iast"), "splitter.output_encoding"
        )

        execution = mapping.get("execution", {})
        if not isinstance(execution, Mapping):
            raise ConfigurationError("execution must be an object")
        warm_repetitions = cls._positive_integer(
            execution.get("warm_repetitions", 1), "execution.warm_repetitions"
        )
        cold_repetitions = cls._positive_integer(
            execution.get("cold_repetitions", 7), "execution.cold_repetitions"
        )
        warmup_cases = cls._string_tuple(
            execution.get("warmup_cases", ["yoga"]), "execution.warmup_cases"
        )
        cold_case = cls._string(execution.get("cold_case"), "execution.cold_case")
        require_release_native = execution.get("require_release_native", True)
        if not isinstance(require_release_native, bool):
            raise ConfigurationError(
                "execution.require_release_native must be a boolean"
            )

        output = mapping.get("output", {})
        if not isinstance(output, Mapping):
            raise ConfigurationError("output must be an object")
        output_path = cls._resolve(
            root,
            cls._string(
                output.get("path", "build/benchmarks/splitter-benchmark.json"),
                "output.path",
            ),
        )
        include_cases = output.get("include_cases", True)
        if not isinstance(include_cases, bool):
            raise ConfigurationError("output.include_cases must be a boolean")

        correctness = mapping.get("correctness", {})
        if not isinstance(correctness, Mapping):
            raise ConfigurationError("correctness must be an object")
        fail_on_mismatch = correctness.get("fail_on_mismatch", True)
        if not isinstance(fail_on_mismatch, bool):
            raise ConfigurationError(
                "correctness.fail_on_mismatch must be a boolean"
            )

        return cls(
            raw=mapping,
            root=root,
            backends=backends,
            reference_backend=reference_backend,
            backend_environment_variable=backend_environment_variable,
            datasets=tuple(datasets),
            extra_cases=tuple(extra_cases),
            length_buckets=tuple(length_buckets),
            limit=limit,
            score=score,
            input_encoding=input_encoding,
            output_encoding=output_encoding,
            warm_repetitions=warm_repetitions,
            cold_repetitions=cold_repetitions,
            warmup_cases=warmup_cases,
            cold_case=cold_case,
            require_release_native=require_release_native,
            output_path=output_path,
            include_cases=include_cases,
            fail_on_mismatch=fail_on_mismatch,
        )

    def length_bucket(self, length: int) -> LengthBucket:
        matches = [bucket for bucket in self.length_buckets if bucket.matches(length)]
        if len(matches) != 1:
            raise ConfigurationError(
                "length %d must match exactly one configured bucket" % length
            )
        return matches[0]

    @staticmethod
    def _validate_buckets(buckets: Sequence[LengthBucket]) -> None:
        names = [bucket.name for bucket in buckets]
        if len(names) != len(set(names)):
            raise ConfigurationError("length bucket names must be unique")
        ordered = sorted(buckets, key=lambda bucket: bucket.minimum)
        for previous, current in zip(ordered, ordered[1:]):
            if previous.maximum is None or current.minimum <= previous.maximum:
                raise ConfigurationError("length buckets overlap")

    @staticmethod
    def _resolve(root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    @staticmethod
    def _string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError("%s must be a non-empty string" % field)
        return value

    @classmethod
    def _string_tuple(cls, value: Any, field: str) -> Tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise ConfigurationError("%s must be a non-empty list" % field)
        return tuple(cls._string(item, field) for item in value)

    @staticmethod
    def _nonnegative_integer(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError("%s must be a non-negative integer" % field)
        return value

    @classmethod
    def _positive_integer(cls, value: Any, field: str) -> int:
        result = cls._nonnegative_integer(value, field)
        if result == 0:
            raise ConfigurationError("%s must be greater than zero" % field)
        return result


@dataclass(frozen=True)
class BenchmarkCase:
    text: str
    categories: Tuple[str, ...]
    dataset_categories: Tuple[str, ...]
    sources: Tuple[str, ...]
    length: int
    length_bucket: str
    focus_repetitions: int


@dataclass(frozen=True)
class BenchmarkCorpus:
    cases: Tuple[BenchmarkCase, ...]
    loaded_records: int
    duplicate_records: int


class CorpusLoader:
    """Load, validate, and deduplicate all configured corpus records."""

    def __init__(self, configuration: BenchmarkConfiguration):
        self.configuration = configuration

    def load(self) -> BenchmarkCorpus:
        merged: Dict[str, Dict[str, Any]] = {}
        loaded_records = 0
        for dataset in self.configuration.datasets:
            payload = self._read_dataset(dataset.path)
            compounds = payload.get("compounds")
            if not isinstance(compounds, Mapping):
                raise ConfigurationError(
                    "%s must contain a compounds object" % dataset.path
                )
            for category in dataset.categories:
                rows = compounds.get(category)
                if not isinstance(rows, list):
                    raise ConfigurationError(
                        "%s has no list for category %s"
                        % (dataset.path, category)
                    )
                for row_index, row in enumerate(rows):
                    if not isinstance(row, Mapping):
                        raise ConfigurationError(
                            "%s category %s row %d must be an object"
                            % (dataset.path, category, row_index)
                        )
                    text = BenchmarkConfiguration._string(
                        row.get("text"),
                        "%s category %s row %d text"
                        % (dataset.path, category, row_index),
                    )
                    source_file = BenchmarkConfiguration._string(
                        row.get("source_file", "unknown"),
                        "%s category %s row %d source_file"
                        % (dataset.path, category, row_index),
                    )
                    self._merge(
                        merged,
                        text=text,
                        category=category,
                        dataset_category="%s:%s" % (dataset.path.name, category),
                        source="%s/%s" % (dataset.path.name, source_file),
                        focus_repetitions=0,
                    )
                    loaded_records += 1

        for extra in self.configuration.extra_cases:
            self._merge(
                merged,
                text=extra.text,
                category=extra.category,
                dataset_category="configuration:%s" % extra.category,
                source=extra.source,
                focus_repetitions=extra.warm_repetitions,
            )
            loaded_records += 1

        cases = []
        for text, value in merged.items():
            length = len(text)
            cases.append(
                BenchmarkCase(
                    text=text,
                    categories=tuple(value["categories"]),
                    dataset_categories=tuple(value["dataset_categories"]),
                    sources=tuple(value["sources"]),
                    length=length,
                    length_bucket=self.configuration.length_bucket(length).name,
                    focus_repetitions=value["focus_repetitions"],
                )
            )
        return BenchmarkCorpus(
            cases=tuple(cases),
            loaded_records=loaded_records,
            duplicate_records=loaded_records - len(cases),
        )

    @staticmethod
    def _read_dataset(path: Path) -> Mapping[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError("Unable to read dataset %s: %s" % (path, error))
        if not isinstance(value, Mapping):
            raise ConfigurationError("Dataset %s must contain a JSON object" % path)
        return value

    @staticmethod
    def _merge(
        merged: Dict[str, Dict[str, Any]],
        *,
        text: str,
        category: str,
        dataset_category: str,
        source: str,
        focus_repetitions: int,
    ) -> None:
        value = merged.setdefault(
            text,
            {
                "categories": [],
                "dataset_categories": [],
                "sources": [],
                "focus_repetitions": 0,
            },
        )
        for key, item in (
            ("categories", category),
            ("dataset_categories", dataset_category),
            ("sources", source),
        ):
            if item not in value[key]:
                value[key].append(item)
        value["focus_repetitions"] = max(
            value["focus_repetitions"], focus_repetitions
        )


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_count: int
    no_split: bool
    error: Optional[str]
    ordered_digest: str
    multiset_digest: str
    winner_digest: str
    candidates: Optional[Tuple[Tuple[str, ...], ...]]

    @classmethod
    def from_candidates(
        cls, candidates: Optional[Sequence[Sequence[str]]]
    ) -> "CandidateSnapshot":
        if candidates is None:
            payload: Any = {"no_split": True}
            digest = JsonDigest.sha256(payload)
            return cls(0, True, None, digest, digest, digest, None)
        canonical = tuple(tuple(token for token in candidate) for candidate in candidates)
        ordered_value = [list(candidate) for candidate in canonical]
        sorted_value = sorted(
            ordered_value,
            key=lambda candidate: JsonDigest.canonical_bytes(candidate),
        )
        winner: Any = ordered_value[0] if ordered_value else {"empty": True}
        return cls(
            candidate_count=len(canonical),
            no_split=False,
            error=None,
            ordered_digest=JsonDigest.sha256(ordered_value),
            multiset_digest=JsonDigest.sha256(sorted_value),
            winner_digest=JsonDigest.sha256(winner),
            candidates=canonical,
        )

    @classmethod
    def from_error(cls, error: str) -> "CandidateSnapshot":
        digest = JsonDigest.sha256({"error": error})
        return cls(0, False, error, digest, digest, digest, None)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateSnapshot":
        raw_candidates = value.get("candidates")
        candidates = (
            None
            if raw_candidates is None
            else tuple(tuple(candidate) for candidate in raw_candidates)
        )
        return cls(
            candidate_count=int(value["candidate_count"]),
            no_split=bool(value["no_split"]),
            error=value.get("error"),
            ordered_digest=str(value["ordered_digest"]),
            multiset_digest=str(value["multiset_digest"]),
            winner_digest=str(value["winner_digest"]),
            candidates=candidates,
        )

    def as_dict(self, include_candidates: bool = True) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "candidate_count": self.candidate_count,
            "no_split": self.no_split,
            "error": self.error,
            "ordered_digest": self.ordered_digest,
            "multiset_digest": self.multiset_digest,
            "winner_digest": self.winner_digest,
        }
        if include_candidates:
            value["candidates"] = (
                None
                if self.candidates is None
                else [list(candidate) for candidate in self.candidates]
            )
        return value


@dataclass(frozen=True)
class Distribution:
    samples: int
    mean: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    maximum: Optional[float]

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "Distribution":
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return cls(0, None, None, None, None)
        return cls(
            samples=len(ordered),
            mean=statistics.mean(ordered),
            p50=cls._percentile(ordered, 0.50),
            p95=cls._percentile(ordered, 0.95),
            maximum=ordered[-1],
        )

    @staticmethod
    def _percentile(ordered: Sequence[float], quantile: float) -> float:
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "mean": self._rounded(self.mean),
            "p50": self._rounded(self.p50),
            "p95": self._rounded(self.p95),
            "max": self._rounded(self.maximum),
        }

    @staticmethod
    def _rounded(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(value, 6)


class ResourceMonitor:
    """Portable best-effort resident-set measurements without a hard dependency."""

    @staticmethod
    def current_rss_mib() -> Optional[float]:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        if sys.platform.startswith("linux"):
            try:
                resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
                return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
            except (OSError, ValueError, IndexError):
                return None
        return None

    @staticmethod
    def peak_rss_mib() -> Optional[float]:
        try:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (ImportError, ValueError):
            return None
        if sys.platform == "darwin":
            return value / (1024 * 1024)
        return value / 1024


class BackendRuntimeInspector:
    """Identify the exact implementation and tokenizer stack being measured."""

    PACKAGE_NAMES = ("process-sanskrit", "numpy", "sentencepiece")

    @classmethod
    def describe(cls, parser: Any) -> Dict[str, Any]:
        backend = parser._backend
        backend_name = backend.name
        backend_module = sys.modules[type(backend).__module__]
        module_path = Path(inspect.getfile(backend_module)).resolve()
        build_profile = "interpreted"
        native = None
        assets: Dict[str, Any]
        if backend_name == "rust":
            from process_sanskrit.splitter import _native

            module_path = Path(_native.__file__).resolve()
            build_profile = str(_native.BUILD_PROFILE)
            native = {
                "asset_schema_version": int(_native.ASSET_SCHEMA_VERSION),
                "sentencepiece_version": str(_native.SENTENCEPIECE_VERSION),
            }
            manifest_path = module_path.parent / "data" / "native" / "native-assets.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assets = {
                "manifest": cls._asset_record(manifest_path),
                "declared": manifest,
            }
        else:
            from process_sanskrit.splitter.data_manager import data_file_path

            assets = {
                name: cls._asset_record(Path(data_file_path(name)).resolve())
                for name in (
                    "forms.trie",
                    "sandhi_rules.zip",
                    "w2v.npz",
                    "sentencepiece.model",
                )
            }
        return {
            "backend": backend_name,
            "module_path": str(module_path),
            "module_sha256": cls._file_sha256(module_path),
            "build_profile": build_profile,
            "packages": {
                name: importlib.metadata.version(name) for name in cls.PACKAGE_NAMES
            },
            "native": native,
            "assets": assets,
        }

    @classmethod
    def _asset_record(cls, path: Path) -> Dict[str, Any]:
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": cls._file_sha256(path),
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class SplitterAdapter:
    """Keep imports lazy so the worker can select its backend first."""

    def __init__(self, configuration: BenchmarkConfiguration):
        from process_sanskrit.splitter import Parser

        self.parser = Parser(
            input_encoding=configuration.input_encoding,
            output_encoding=configuration.output_encoding,
            score=configuration.score,
        )
        self.limit = configuration.limit
        self._runtime: Optional[Dict[str, Any]] = None

    def runtime_metadata(self) -> Dict[str, Any]:
        if self._runtime is None:
            self._runtime = BackendRuntimeInspector.describe(self.parser)
        return dict(self._runtime)

    def split(self, text: str) -> Tuple[CandidateSnapshot, Tuple[str, ...]]:
        captured = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                splits = self.parser.split(text, limit=self.limit)
            captured = [str(item.message) for item in caught]
            if splits is None:
                return CandidateSnapshot.from_candidates(None), tuple(captured)
            candidates = []
            for split in splits:
                value = ast.literal_eval(str(split))
                if not isinstance(value, list) or not all(
                    isinstance(token, str) for token in value
                ):
                    raise TypeError("Parser split output is not a list of strings")
                candidates.append(value)
            return CandidateSnapshot.from_candidates(candidates), tuple(captured)
        except Exception as error:  # The report records per-input failures.
            message = "%s: %s" % (type(error).__name__, error)
            return CandidateSnapshot.from_error(message), tuple(captured)


@dataclass
class CaseMeasurement:
    case: BenchmarkCase
    timings_ms: List[float]
    snapshot: CandidateSnapshot
    deterministic: bool
    warnings: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.case.text,
            "length": self.case.length,
            "length_bucket": self.case.length_bucket,
            "categories": list(self.case.categories),
            "dataset_categories": list(self.case.dataset_categories),
            "sources": list(self.case.sources),
            "timing_ms": Distribution.from_values(self.timings_ms).as_dict(),
            "deterministic": self.deterministic,
            "warnings": list(self.warnings),
            "result": self.snapshot.as_dict(include_candidates=True),
        }


class WarmBenchmarkWorker:
    def __init__(
        self,
        configuration: BenchmarkConfiguration,
        corpus: BenchmarkCorpus,
        backend: str,
    ):
        self.configuration = configuration
        self.corpus = corpus
        self.backend = backend

    def run(self) -> Dict[str, Any]:
        rss_before_import = ResourceMonitor.current_rss_mib()
        initialized = time.perf_counter_ns()
        adapter = SplitterAdapter(self.configuration)
        initialization_ms = (time.perf_counter_ns() - initialized) / 1_000_000
        rss_after_initialization = ResourceMonitor.current_rss_mib()

        for text in self.configuration.warmup_cases:
            snapshot, _ = adapter.split(text)
            if snapshot.error:
                raise RuntimeError("Warmup failed for %r: %s" % (text, snapshot.error))

        rss_before_measurement = ResourceMonitor.current_rss_mib()
        measurements = [self._measure_case(adapter, case) for case in self.corpus.cases]
        focus = self._measure_focus_cases(adapter)
        rss_after_measurement = ResourceMonitor.current_rss_mib()

        return {
            "backend": self.backend,
            "runtime": adapter.runtime_metadata(),
            "initialization_ms": round(initialization_ms, 6),
            "timing_ms": Distribution.from_values(
                timing
                for measurement in measurements
                for timing in measurement.timings_ms
            ).as_dict(),
            "by_length": self._group(measurements, "length"),
            "by_category": self._group(measurements, "category"),
            "by_dataset_category": self._group(
                measurements, "dataset_category"
            ),
            "rss_mib": {
                "before_import": self._round(rss_before_import),
                "after_initialization": self._round(rss_after_initialization),
                "before_measurement": self._round(rss_before_measurement),
                "after_measurement": self._round(rss_after_measurement),
                "measurement_delta": self._delta(
                    rss_before_measurement, rss_after_measurement
                ),
                "peak": self._round(ResourceMonitor.peak_rss_mib()),
            },
            "correctness": self._correctness(measurements),
            "focus_cases": focus,
            "cases": [measurement.as_dict() for measurement in measurements],
        }

    def _measure_case(
        self, adapter: SplitterAdapter, case: BenchmarkCase
    ) -> CaseMeasurement:
        timings = []
        first_snapshot = None
        first_warnings: Tuple[str, ...] = ()
        deterministic = True
        for _ in range(self.configuration.warm_repetitions):
            started = time.perf_counter_ns()
            snapshot, caught = adapter.split(case.text)
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
            if first_snapshot is None:
                first_snapshot = snapshot
                first_warnings = caught
            elif (
                snapshot.ordered_digest != first_snapshot.ordered_digest
                or snapshot.error != first_snapshot.error
                or caught != first_warnings
            ):
                deterministic = False
        assert first_snapshot is not None
        return CaseMeasurement(
            case=case,
            timings_ms=timings,
            snapshot=first_snapshot,
            deterministic=deterministic,
            warnings=first_warnings,
        )

    def _measure_focus_cases(self, adapter: SplitterAdapter) -> List[Dict[str, Any]]:
        values = []
        for case in self.corpus.cases:
            if case.focus_repetitions == 0:
                continue
            timings = []
            first_snapshot = None
            deterministic = True
            for _ in range(case.focus_repetitions):
                started = time.perf_counter_ns()
                snapshot, _ = adapter.split(case.text)
                timings.append((time.perf_counter_ns() - started) / 1_000_000)
                if first_snapshot is None:
                    first_snapshot = snapshot
                elif snapshot.ordered_digest != first_snapshot.ordered_digest:
                    deterministic = False
            assert first_snapshot is not None
            values.append(
                {
                    "text": case.text,
                    "repetitions": case.focus_repetitions,
                    "timing_ms": Distribution.from_values(timings).as_dict(),
                    "deterministic": deterministic,
                    "result": first_snapshot.as_dict(include_candidates=True),
                }
            )
        return values

    @staticmethod
    def _group(
        measurements: Sequence[CaseMeasurement], group_type: str
    ) -> Dict[str, Any]:
        groups: Dict[str, Dict[str, Any]] = {}
        for measurement in measurements:
            if group_type == "length":
                names = (measurement.case.length_bucket,)
            elif group_type == "category":
                names = measurement.case.categories
            else:
                names = measurement.case.dataset_categories
            for name in names:
                group = groups.setdefault(name, {"cases": set(), "timings": []})
                group["cases"].add(measurement.case.text)
                group["timings"].extend(measurement.timings_ms)
        return {
            name: {
                "unique_cases": len(group["cases"]),
                "timing_ms": Distribution.from_values(group["timings"]).as_dict(),
            }
            for name, group in sorted(groups.items())
        }

    @staticmethod
    def _correctness(measurements: Sequence[CaseMeasurement]) -> Dict[str, Any]:
        ordered_values = {}
        multiset_values = {}
        winner_values = {}
        errors = []
        no_splits = []
        nondeterministic = []
        for measurement in measurements:
            text = measurement.case.text
            snapshot = measurement.snapshot
            ordered_values[text] = snapshot.ordered_digest
            multiset_values[text] = snapshot.multiset_digest
            winner_values[text] = snapshot.winner_digest
            if snapshot.error:
                errors.append({"text": text, "error": snapshot.error})
            if snapshot.no_split:
                no_splits.append(text)
            if not measurement.deterministic:
                nondeterministic.append(text)
        return {
            "case_count": len(measurements),
            "error_count": len(errors),
            "no_split_count": len(no_splits),
            "nondeterministic_count": len(nondeterministic),
            "ordered_suite_digest": JsonDigest.sha256(ordered_values),
            "candidate_multiset_suite_digest": JsonDigest.sha256(multiset_values),
            "winner_suite_digest": JsonDigest.sha256(winner_values),
            "errors": errors,
            "no_splits": no_splits,
            "nondeterministic": nondeterministic,
        }

    @staticmethod
    def _round(value: Optional[float]) -> Optional[float]:
        return None if value is None else round(value, 6)

    @classmethod
    def _delta(
        cls, before: Optional[float], after: Optional[float]
    ) -> Optional[float]:
        return None if before is None or after is None else cls._round(after - before)


class ColdBenchmarkWorker:
    def __init__(self, configuration: BenchmarkConfiguration, backend: str):
        self.configuration = configuration
        self.backend = backend

    def run(self) -> Dict[str, Any]:
        rss_before = ResourceMonitor.current_rss_mib()
        started = time.perf_counter_ns()
        adapter = SplitterAdapter(self.configuration)
        snapshot, caught = adapter.split(self.configuration.cold_case)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "backend": self.backend,
            "runtime": adapter.runtime_metadata(),
            "text": self.configuration.cold_case,
            "timing_ms": round(elapsed_ms, 6),
            "rss_mib": {
                "before_import": WarmBenchmarkWorker._round(rss_before),
                "after_split": WarmBenchmarkWorker._round(
                    ResourceMonitor.current_rss_mib()
                ),
                "peak": WarmBenchmarkWorker._round(ResourceMonitor.peak_rss_mib()),
            },
            "warnings": list(caught),
            "result": snapshot.as_dict(include_candidates=True),
        }


class CorrectnessComparator:
    """Compare observable candidates, order, winner, and failures by input."""

    def __init__(
        self,
        reference: Mapping[str, CandidateSnapshot],
        target: Mapping[str, CandidateSnapshot],
    ):
        self.reference = reference
        self.target = target

    def summarize(self) -> Dict[str, Any]:
        texts = sorted(set(self.reference) | set(self.target))
        multiset_matches = 0
        ordered_matches = 0
        winner_matches = 0
        mismatches = []
        for text in texts:
            reference = self.reference.get(text)
            target = self.target.get(text)
            error_free = bool(
                reference
                and target
                and reference.error is None
                and target.error is None
            )
            multiset = bool(
                error_free
                and reference.multiset_digest == target.multiset_digest
            )
            ordered = bool(
                error_free
                and reference.ordered_digest == target.ordered_digest
            )
            winner = bool(
                error_free
                and reference.winner_digest == target.winner_digest
            )
            multiset_matches += int(multiset)
            ordered_matches += int(ordered)
            winner_matches += int(winner)
            if not (multiset and ordered and winner):
                mismatches.append(
                    {
                        "text": text,
                        "candidate_multiset_match": multiset,
                        "ordered_candidates_match": ordered,
                        "winner_match": winner,
                        "reference_error": None if reference is None else reference.error,
                        "target_error": None if target is None else target.error,
                    }
                )
        case_count = len(texts)
        behavioral_parity = (
            multiset_matches == case_count and winner_matches == case_count
        )
        ordered_parity = ordered_matches == case_count
        return {
            "case_count": case_count,
            "candidate_multiset_matches": multiset_matches,
            "ordered_candidate_matches": ordered_matches,
            "winner_matches": winner_matches,
            "behavioral_parity": behavioral_parity,
            "ordered_parity": ordered_parity,
            "exact_parity": behavioral_parity and ordered_parity,
            "mismatches": mismatches,
        }


class CorrectnessGate:
    """Require parity plus error-free, deterministic benchmark execution."""

    @classmethod
    def passes(
        cls,
        report: Mapping[str, Any],
        *,
        require_release_native: bool = True,
    ) -> bool:
        return not cls.failures(
            report, require_release_native=require_release_native
        )

    @staticmethod
    def failures(
        report: Mapping[str, Any],
        *,
        require_release_native: bool = True,
    ) -> Tuple[str, ...]:
        failures = []
        for backend, comparison in report["comparisons"].items():
            if not comparison["behavioral_parity"]:
                failures.append("%s behavioral mismatch" % backend)

        for backend, result in report["backends"].items():
            if not result["runtime_consistent"]:
                failures.append("%s cold/warm runtime mismatch" % backend)

            cold = result["cold"]
            if cold["result"]["error"]:
                failures.append("%s cold error" % backend)
            if not cold["deterministic"]:
                failures.append("%s cold nondeterminism" % backend)
            if not cold["runtime_consistent"]:
                failures.append("%s cold runtime mismatch" % backend)

            warm = result["warm"]
            correctness = warm["correctness"]
            if correctness["error_count"]:
                failures.append("%s warm errors" % backend)
            if correctness["nondeterministic_count"]:
                failures.append("%s warm nondeterminism" % backend)
            for focus in warm["focus_cases"]:
                if focus["result"]["error"]:
                    failures.append("%s focus error" % backend)
                if not focus["deterministic"]:
                    failures.append("%s focus nondeterminism" % backend)
            if require_release_native and backend == "rust":
                profiles = (
                    cold["runtime"]["build_profile"],
                    warm["runtime"]["build_profile"],
                )
                if any(profile != "release" for profile in profiles):
                    failures.append("rust backend is not release-built")
        return tuple(failures)


class BackendCoordinator:
    """Run cold and warm workers in clean backend-specific processes."""

    def __init__(
        self, configuration: BenchmarkConfiguration, configuration_path: Path
    ):
        self.configuration = configuration
        self.configuration_path = configuration_path

    def run(self, backend: str) -> Dict[str, Any]:
        cold_samples = [self._invoke_worker(backend, "cold") for _ in range(
            self.configuration.cold_repetitions
        )]
        warm = self._invoke_worker(backend, "warm")
        cold_digests = {
            sample["result"]["ordered_digest"] for sample in cold_samples
        }
        cold_runtime_digests = {
            JsonDigest.sha256(sample["runtime"]) for sample in cold_samples
        }
        runtime_consistent = RuntimeProvenance.consistent(
            tuple(sample["runtime"] for sample in cold_samples)
            + (warm["runtime"],)
        )
        compact_samples = [
            {
                "timing_ms": sample["timing_ms"],
                "parent_wall_ms": sample["parent_wall_ms"],
                "rss_mib": sample["rss_mib"],
                "warnings": sample["warnings"],
                "ordered_digest": sample["result"]["ordered_digest"],
                "error": sample["result"]["error"],
                "no_split": sample["result"]["no_split"],
                "runtime": sample["runtime"],
            }
            for sample in cold_samples
        ]
        return {
            "runtime_consistent": runtime_consistent,
            "cold": {
                "text": self.configuration.cold_case,
                "repetitions": len(cold_samples),
                "timing_ms": Distribution.from_values(
                    sample["timing_ms"] for sample in cold_samples
                ).as_dict(),
                "parent_wall_ms": Distribution.from_values(
                    sample["parent_wall_ms"] for sample in cold_samples
                ).as_dict(),
                "peak_rss_mib": Distribution.from_values(
                    sample["rss_mib"]["peak"]
                    for sample in cold_samples
                    if sample["rss_mib"]["peak"] is not None
                ).as_dict(),
                "deterministic": len(cold_digests) == 1,
                "runtime_consistent": len(cold_runtime_digests) == 1,
                "runtime": cold_samples[0]["runtime"],
                "result": cold_samples[0]["result"],
                "samples": compact_samples,
            },
            "warm": warm,
        }

    def _invoke_worker(self, backend: str, worker: str) -> Dict[str, Any]:
        environment = os.environ.copy()
        environment[self.configuration.backend_environment_variable] = backend
        environment[WORKER_ENV] = worker
        environment[WORKER_BACKEND_ENV] = backend
        environment[WORKER_CONFIG_ENV] = str(self.configuration_path)
        started = time.perf_counter_ns()
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
            cwd=self.configuration.root,
            env=environment,
            capture_output=True,
            text=True,
        )
        parent_wall_ms = (time.perf_counter_ns() - started) / 1_000_000
        if completed.returncode != 0:
            raise RuntimeError(
                "%s %s worker failed:\n%s"
                % (backend, worker, completed.stderr.strip())
            )
        try:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            raise RuntimeError(
                "%s %s worker returned invalid JSON: %s"
                % (backend, worker, completed.stdout)
            ) from error
        result["parent_wall_ms"] = round(parent_wall_ms, 6)
        return result


class BenchmarkReport:
    def __init__(
        self,
        configuration: BenchmarkConfiguration,
        corpus: BenchmarkCorpus,
        backend_results: Mapping[str, Dict[str, Any]],
    ):
        self.configuration = configuration
        self.corpus = corpus
        self.backend_results = dict(backend_results)

    def build(self) -> Dict[str, Any]:
        comparisons = self._comparisons()
        backend_results = self.backend_results
        if not self.configuration.include_cases:
            backend_results = json.loads(json.dumps(backend_results))
            for result in backend_results.values():
                result["warm"].pop("cases", None)
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": self.configuration.digest,
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "corpus": {
                "loaded_records": self.corpus.loaded_records,
                "unique_cases": len(self.corpus.cases),
                "duplicate_records": self.corpus.duplicate_records,
                "dataset_records": self.corpus.loaded_records
                - len(self.configuration.extra_cases),
                "extra_records": len(self.configuration.extra_cases),
            },
            "reference_backend": self.configuration.reference_backend,
            "backends": backend_results,
            "comparisons": comparisons,
        }

    def _comparisons(self) -> Dict[str, Any]:
        reference_name = self.configuration.reference_backend
        reference = self._snapshots(self.backend_results[reference_name])
        return {
            backend: CorrectnessComparator(
                reference, self._snapshots(result)
            ).summarize()
            for backend, result in self.backend_results.items()
            if backend != reference_name
        }

    @staticmethod
    def _snapshots(result: Mapping[str, Any]) -> Dict[str, CandidateSnapshot]:
        return {
            case["text"]: CandidateSnapshot.from_dict(case["result"])
            for case in result["warm"]["cases"]
        }


class BenchmarkSummary:
    """Small JSON stdout payload pointing to the complete on-disk report."""

    @staticmethod
    def build(report: Mapping[str, Any], output_path: Path) -> Dict[str, Any]:
        backends = {}
        for backend, result in report["backends"].items():
            correctness = result["warm"]["correctness"]
            focus_cases = [
                {
                    "text": case["text"],
                    "timing_ms": case["timing_ms"],
                    "ordered_digest": case["result"]["ordered_digest"],
                    "error": case["result"]["error"],
                }
                for case in result["warm"]["focus_cases"]
            ]
            backends[backend] = {
                "runtime": {
                    "cold": result["cold"]["runtime"],
                    "warm": result["warm"]["runtime"],
                    "cold_consistent": result["cold"]["runtime_consistent"],
                    "consistent": result["runtime_consistent"],
                },
                "cold_timing_ms": result["cold"]["timing_ms"],
                "warm_timing_ms": result["warm"]["timing_ms"],
                "warm_peak_rss_mib": result["warm"]["rss_mib"]["peak"],
                "correctness": {
                    key: correctness[key]
                    for key in (
                        "case_count",
                        "error_count",
                        "no_split_count",
                        "nondeterministic_count",
                        "ordered_suite_digest",
                        "candidate_multiset_suite_digest",
                        "winner_suite_digest",
                    )
                },
                "focus_cases": focus_cases,
            }
        return {
            "output_path": str(output_path),
            "corpus": report["corpus"],
            "backends": backends,
            "comparisons": report["comparisons"],
        }


class BenchmarkApplication:
    def run(self, arguments: Optional[Sequence[str]] = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--config",
            type=Path,
            default=DEFAULT_CONFIG,
            help="JSON configuration file (default: %(default)s)",
        )
        parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        options = parser.parse_args(arguments)
        if options.worker:
            return self._run_worker()

        configuration_path = options.config.resolve()
        configuration = BenchmarkConfiguration.load(configuration_path)
        corpus = CorpusLoader(configuration).load()
        coordinator = BackendCoordinator(configuration, configuration_path)
        backend_results = {
            backend: coordinator.run(backend) for backend in configuration.backends
        }
        report = BenchmarkReport(configuration, corpus, backend_results).build()
        configuration.output_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        configuration.output_path.write_text(encoded + "\n", encoding="utf-8")
        print(
            json.dumps(
                BenchmarkSummary.build(report, configuration.output_path),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        failed_correctness = not CorrectnessGate.passes(
            report,
            require_release_native=configuration.require_release_native,
        )
        return int(configuration.fail_on_mismatch and failed_correctness)

    @staticmethod
    def _run_worker() -> int:
        config_value = os.environ.get(WORKER_CONFIG_ENV)
        backend = os.environ.get(WORKER_BACKEND_ENV)
        worker = os.environ.get(WORKER_ENV)
        if not config_value or not backend or worker not in ("cold", "warm"):
            raise RuntimeError("Incomplete internal benchmark worker environment")
        configuration = BenchmarkConfiguration.load(Path(config_value), ROOT)
        if backend not in configuration.backends:
            raise ConfigurationError("Worker backend is not configured: %s" % backend)
        if worker == "cold":
            result = ColdBenchmarkWorker(configuration, backend).run()
        else:
            corpus = CorpusLoader(configuration).load()
            result = WarmBenchmarkWorker(configuration, corpus, backend).run()
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return 0


if __name__ == "__main__":
    raise SystemExit(BenchmarkApplication().run())
