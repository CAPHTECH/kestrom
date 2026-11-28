# Report format

Every report is one JSON object written with `indent=1` and no times. The keys present depend on the command.

| Key | Written by | Content |
|---|---|---|
| `original`, `edited` | `range`, `logits`, `measure`, `diff` | `{path in the folder: SHA-256}` of every file under each model's folder (a GGUF file by its name) |
| `weights` | `range`, `logits` | `output layer` (tensor name), `tied to the embedding` (true, only when it is), `files`, `files equal but for the changed rows`, `files not equal` (paths), `changed rows` (token ids) |
| `inputs` | `logits`, `measure` | `{file name: SHA-256}` of the inputs |
| `forced` | `logits` | Totals: `sequences`, `positions`, `original_equal`, `other_equal`, `positions_other_equal`, `positions_top_changed`, `answer_positions`, `answer_edited_top_changed`, `answer_max_changed_logit`; tied to the embedding, also `sequences_outside` (G1 not checked; `other_equal` counts only the others) |
| `range` | `compare` | `{"events": ...}` or `{range report name: SHA-256}` |
| `prompts` | `compare` with a range tied to the embedding | `{file name: SHA-256}` of PROMPTS.json |
| `generations` | `compare`, `measure` | `{file name: SHA-256}` of the greedy outputs |
| `greedy` | `compare` | Totals: `prompts`, `original twice the same`, `same`, `same or first difference in the range` (over the prompts checked), tied to the embedding `prompts with a changed id` (not checked), and the events or range tokens in the edited outputs |
| `measure` | `measure` | Totals: `sequences`, `original_equal`, `positions`, `positions_all_equal`, `positions_top_changed`, `sequences_all_equal`, `max_abs_top_logit_difference` |
| `measure greedy` | `measure` | `prompts`, `original twice the same`, `same` |
| `diff` | `diff` | Totals: `tensors`, `tensors the same`, `tensors changed`, `elements`, `changed elements`, `tensors changed with no element count` (quantized blocks), `other files changed` |
| `tensors` | `diff` | Per changed tensor: `change` (`values`, `added`, `removed`, `dtype or shape`), and for values `dtype`, `shape`, `rows`, `changed rows` (spans), `changed row count`, `elements`, `changed elements`; for the others `original` and `edited` (`dtype` and `shape`, or null for the side without the tensor) |
| `files` | `diff` | Other files (and `__metadata__`): `added`, `removed` or `changed`; `__container__`: `anomaly` when the edited model's container anomalies differ from the original's |
| `metadata_diff` | `diff`, when the metadata differ | `added`, `removed`, `changed`: the metadata keys (GGUF key-value fields) |
| `container_anomalies` | `diff`, when the edited model has any | The edited model's `anomalies`, as in `container` |
| `model` | `container` | `{path in the folder: SHA-256}` of every file under the model's folder (a GGUF file by its name) |
| `container` | `container` | `format` (`safetensors` or `gguf`), `status` (`PASS` or `FLAGGED`), `files_scanned`, `files_clean`, `alignment` (GGUF), `anomalies` (each with `file` and `kind`: `trailing_bytes`, `inter_tensor_gap`, `overlap`, `out_of_bounds`, `empty_tensors`; with the offsets, and for unreferenced bytes `length`, `nonzero_bytes`, `sha256`, `head_hex`), `chat_templates` (`{key: {length, sha256, text}}`) |
| `declared`, `declaration file` | `diff` with a declaration | `declaration`, `changes`, `inside`, `outside` (names); the declaration's SHA-256 |
| `rows` | `logits` | One entry per sequence: the keys of the input but `ids` and `tokens`, `positions`, `original_equal`, `other_equal`, `positions_other_equal`, `positions_top_changed`, `answer_positions`, `answer_edited_top_changed`, `answer_max_changed`; tied to the embedding, `changed_id_in_input` |
| `rows` | `measure` | One entry per sequence: the keys of the input but `ids` and `tokens`, `positions`, `original_equal`, `positions_all_equal`, `positions_top_changed`, `max_abs_top_difference` |
| `rows` | `compare` | One entry per prompt: `id`, `original twice the same`, `same`, `first difference`, `in the range`; tied to the embedding, `changed id in the prompt` |

The verdict is computed from these keys (`kestrom_verify.report.verdict`), so `summary` gives the same PASS / FAIL as the command did.

## Acceptance audit report (`kestrom audit`)

`kestrom audit` (`edit/`) writes deterministic, timestamp-free `audit.json`, `audit.html`, and `audit.sarif` alongside `container.json` (the report of `kestrom-verify container`), `mirror.json` (safetensors only) and `diff.json` (when `BASE_DIR` is given). `verdict` is `FLAGGED` (exiting 1) when `container`, `diff` or `mirror` is `FLAGGED`; otherwise `PASS` only when the container is clean, tier 2 `diff` + `DECLARED.json` passes D and `mirror` ran and flags nothing, and `NO_FINDINGS` when no declaration was checked or `mirror` was `SKIPPED` (a GGUF file).

| Key | Written by | Content |
|---|---|---|
| `schema` | `audit` | `"kestrom-audit-v1"` |
| `verdict` | `audit` | `"PASS"`, `"NO_FINDINGS"`, or `"FLAGGED"` |
| `model` | `audit` | `{"path": ..., "files": {path: SHA-256}}` for `MODEL_DIR` |
| `base` | `audit` | `{"path": ..., "files": {path: SHA-256}}` for `BASE_DIR`, or `null` |
| `checks` | `audit` | Ordered list of checks (`container` at `tier` 2 with `status` `PASS` / `FLAGGED`, `format`, `files_scanned`, `files_clean`, `anomalies`, `chat_templates`; `diff` at `tier` 2 when `BASE_DIR` is given, with `status` `PASS` / `MEASURED` / `FLAGGED`, `summary`, `outside`, and `metadata_diff` when the metadata differ; `mirror` at `tier` 3 with `status` `NO_FINDINGS` / `FLAGGED`, `quantizer`, `tensors_scanned`, `flagged`, `reasons`, `weight_summary`, or `SKIPPED` with `reason` for a GGUF file) |
| `reproduce` | `audit` | Shell commands to reproduce the underlying `kestrom-verify container`, `kestrom-verify diff` and `kestrom inspect mirror` runs |
| `limitations` | `audit` | Explicit tier scope statements (tier 2 `diff` does not bound outputs; tier 3 `mirror` is statistical evidence, not mathematical proof) |

## Operational KV reports (`kestrom kv from-table`, `verify-chat`, `check-template`, `ingest`)

These commands write deterministic, timestamp-free JSON (`indent=1`, exit `1` on `FAIL` or incompatible templates):

| Schema | Written by | Key fields |
|---|---|---|
| `"kestrom-kv-table-v1"` | `kv from-table` | `source` (`{file: sha256}`), `outputs` (`rules.dsl`, `facts.json`, `generate.dsl`, `regression.json` hashes), `totals` (`entries`, `positive`, `positive_exact`, `outside`, `outside_quiet`, `collisions`), `collisions`, `verdict` (`"PASS"` \| `"FAIL"`) |
| `"kestrom-kv-verify-chat-v1"` | `kv verify-chat` | `bank` (`manifest.json`, `rules.dsl` hashes), `turns_file`, `settings` (`max_tokens`, `max_match`, `prefix_mode`), `totals` (`turns`, `kv_matched_turns`, `kv_none_turns`, `kv_none_history_pure`, `kv_none_u0_equal`, `kv_none_bit_identical`), `rows`, `verdict` (`"PASS"` \| `"FAIL"`) |
| `"kestrom-kv-template-v1"` | `kv check-template` | `model`, `system` (`ok`, `strip_prefix`, `prefix_len`, `error`), `prefix_turn` (same fields), `recommended_mode` (`"system"` \| `"prefix-turn"` \| `null`) |
| `"kestrom-kv-ingest-v1"` | `kv ingest` (`ingest_report.json`) | `mode`, `docs`, `reused`, `extracted`, `valid_count`, `outside_default_hits`, `verdict` (`"PASS"` \| `"FAIL"`) |

`kv ingest` prints its verdict and exits 0 on `FAIL` as well.
