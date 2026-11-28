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
| `files` | `diff` | Other files (and `__metadata__`): `added`, `removed` or `changed` |
| `declared`, `declaration file` | `diff` with a declaration | `declaration`, `changes`, `inside`, `outside` (names); the declaration's SHA-256 |
| `rows` | `logits` | One entry per sequence: the keys of the input but `ids` and `tokens`, `positions`, `original_equal`, `other_equal`, `positions_other_equal`, `positions_top_changed`, `answer_positions`, `answer_edited_top_changed`, `answer_max_changed`; tied to the embedding, `changed_id_in_input` |
| `rows` | `measure` | One entry per sequence: the keys of the input but `ids` and `tokens`, `positions`, `original_equal`, `positions_all_equal`, `positions_top_changed`, `max_abs_top_difference` |
| `rows` | `compare` | One entry per prompt: `id`, `original twice the same`, `same`, `first difference`, `in the range`; tied to the embedding, `changed id in the prompt` |

The verdict is computed from these keys (`kestrom_verify.report.verdict`), so `summary` gives the same PASS / FAIL as the command did.

## Acceptance audit report (`kestrom audit`)

`kestrom audit` (`edit/`) writes deterministic, timestamp-free `audit.json`, `audit.html`, and `audit.sarif` alongside `mirror.json` (and `diff.json` when `BASE_DIR` is given). `verdict` is `PASS` only when tier 2 `diff` + `DECLARED.json` passes D and `mirror` flags nothing, `NO_FINDINGS` when no declaration was checked and `mirror` flags nothing, and `FLAGGED` (exiting 1) otherwise.

| Key | Written by | Content |
|---|---|---|
| `schema` | `audit` | `"kestrom-audit-v1"` |
| `verdict` | `audit` | `"PASS"`, `"NO_FINDINGS"`, or `"FLAGGED"` |
| `model` | `audit` | `{"path": ..., "files": {path: SHA-256}}` for `MODEL_DIR` |
| `base` | `audit` | `{"path": ..., "files": {path: SHA-256}}` for `BASE_DIR`, or `null` |
| `checks` | `audit` | Ordered list of checks (`diff` at `tier` 2 when `BASE_DIR` is given, with `status` `PASS` / `MEASURED` / `FLAGGED`, `summary`, `outside`; `mirror` at `tier` 3 with `status` `NO_FINDINGS` / `FLAGGED`, `quantizer`, `tensors_scanned`, `flagged`, `reasons`, `weight_summary`) |
| `reproduce` | `audit` | Shell commands to reproduce the underlying `kestrom-verify diff` and `kestrom inspect mirror` runs |
| `limitations` | `audit` | Explicit tier scope statements (tier 2 `diff` does not bound outputs; tier 3 `mirror` is statistical evidence, not mathematical proof) |

## Bridge diagnostic (`kestrom bridge diagnose ... OUT.json`)

Written with `indent=1` and sorted keys, no times. It is a heuristic check, not a guarantee (`docs/guarantees.md`).

| Key | Content |
|---|---|
| `status` | `"OK"` with no warnings, else `"WARN"` |
| `layers` | One entry per mapped recipient layer: `donor_layer`, `recip_layer`, `dir_retention`, `cond_retention`, `threshold`; where the layer has a cond, `calib` (`n`, `score_mean`, `score_std`, `fire_rate`, `anchor_coverage`) over CALIB_PROMPTS, and with PROD_PROMPTS also `prod` (the same keys) and `shift` (`cohen_d`, 999.0 when unbounded; `std_ratio`, `fire_rate_shift`, `ks_stat`) |
| `warnings` | The warnings the command printed |

## Operational KV reports (`kestrom kv from-table`, `verify-chat`, `check-template`, `ingest`)

These commands write deterministic, timestamp-free JSON (`indent=1`, exit `1` on `FAIL` or incompatible templates):

| Schema | Written by | Key fields |
|---|---|---|
| `"kestrom-kv-table-v1"` | `kv from-table` | `source` (`{file: sha256}`), `outputs` (`rules.dsl`, `facts.json`, `generate.dsl`, `regression.json` hashes), `totals` (`entries`, `positive`, `positive_exact`, `outside`, `outside_quiet`, `collisions`), `collisions`, `verdict` (`"PASS"` \| `"FAIL"`) |
| `"kestrom-kv-verify-chat-v1"` | `kv verify-chat` | `bank` (`manifest.json`, `rules.dsl` hashes), `turns_file`, `settings` (`max_tokens`, `max_match`, `prefix_mode`), `totals` (`turns`, `kv_matched_turns`, `kv_none_turns`, `kv_none_history_pure`, `kv_none_u0_equal`, `kv_none_bit_identical`), `rows`, `verdict` (`"PASS"` \| `"FAIL"`) |
| `"kestrom-kv-template-v1"` | `kv check-template` | `model`, `system` (`ok`, `strip_prefix`, `prefix_len`, `error`), `prefix_turn` (same fields), `recommended_mode` (`"system"` \| `"prefix-turn"` \| `null`) |
| `"kestrom-kv-ingest-v1"` | `kv ingest` (`ingest_report.json`) | `mode`, `docs`, `reused`, `extracted`, `valid_count`, `outside_default_hits`, `verdict` (`"PASS"` \| `"FAIL"`) |

`kv ingest` prints its verdict and exits 0 on `FAIL` as well.
