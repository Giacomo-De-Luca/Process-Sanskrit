# Local analysis cache

Process-Sanskrit keeps expensive sandhi and morphology analyses in a separate,
writable SQLite database. The downloaded dictionary/inflection database remains
read-only and immutable. SQLite is disk-backed: only a small, bounded page cache
is retained in RAM.

## Behaviour

Caching is enabled by default. It applies only when `process()` reaches the
sandhi/compound fallback; exact dictionary and ordinary morphology lookups are
not copied into the cache. A hit skips both sandhi parsing and `inflect()`, then
runs the current dictionary selection and output cleanup so `mode` and selected
dictionaries remain live.

Use the per-call override when persistence is unwanted:

```python
from process_sanskrit import process

process("cittavṛttinirodhaḥ")                # configured default
process("cittavṛttinirodhaḥ", cached=True)   # force cache use
process("cittavṛttinirodhaḥ", cached=False)  # no cache file, read, or write
```

Direct `sandhi_splitter()` and `hybrid_sandhi_splitter()` calls preserve their
existing `cached=False` default. Passing `cached=True` caches their chosen
non-detailed result. Detailed calls always recompute because candidate lists are
not stored.

## Configuration

Set configuration before the first cached request in a process:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PROCESS_SANSKRIT_CACHE_ENABLED` | `true` | Enables the process-level cache. |
| `PROCESS_SANSKRIT_CACHE_RETENTION` | `prune` | `prune` or `keep_all`. |
| `PROCESS_SANSKRIT_CACHE_MAX_AGE_DAYS` | `90` | Inactivity threshold in `prune` mode. |
| `PROCESS_SANSKRIT_CACHE_PATH` | platform data directory | Overrides the SQLite file path. |

Default paths are:

- macOS: `~/Library/Application Support/process-sanskrit/analysis-cache.sqlite3`
- Linux: `$XDG_DATA_HOME/process-sanskrit/analysis-cache.sqlite3`, falling back
  to `~/.local/share/process-sanskrit/analysis-cache.sqlite3`
- Windows: `%LOCALAPPDATA%/process-sanskrit/analysis-cache.sqlite3`

New directories and files are private to the current user where the operating
system supports POSIX permissions. Cache failures are logged once and fail open:
analysis continues without persistence.

## Retention

`prune` removes entries whose last use is strictly older than the configured
age. Access timestamps are updated at most once per day to keep hits mostly
read-only. Cleanup is opportunistic, indexed, and bounded; it uses passive WAL
checkpoints and incremental vacuum rather than blocking full vacuum operations.

`keep_all` never deletes analysis records. Changing from `keep_all` to `prune`
makes old inactive records eligible for deletion; changing back cannot restore
records already removed.

The cache uses a small SQLAlchemy `QueuePool` with one retained connection, one
overflow connection, and a 2 MiB SQLite page cache per connection. Each web
worker owns its own engine; inherited connections and sessions are discarded
after `fork()`.

## Stored data and ML use

Each row is one canonical prediction for a normalized IAST input, analysis kind,
splitter settings, algorithm signature, and lexicon fingerprint. It stores:

- the first raw input and normalized input;
- the chosen split and grammar/morphology result;
- score, subscores, result source, status, and computation time;
- creation and last-access timestamps.

Dictionary HTML, full candidate lists, request-event history, and human labels
are not stored. Algorithm or lexicon changes produce a new key rather than
overwriting an old prediction. In `keep_all` mode this creates a useful
de-duplicated prediction corpus, but it is not ground-truth training data and
does not preserve query frequency or every original-script variant.

Payloads use a strict tagged-JSON format that preserves list/tuple distinctions;
pickle is never used. Schema migrations use SQLite `user_version`, independently
from the analysis signature used for result invalidation.

