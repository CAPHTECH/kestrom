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
