# Kestrom

Two packages. `verify/` is `kestrom-verify`, the checker (Apache-2.0). `edit/` is `kestrom`, the editing tools (PolyForm, or a
commercial license). `edit` imports `kestrom_verify`; `verify` never imports `kestrom`, so the checker stays free of the editing
license (the Stop hook checks it).

## Commands

- Tests: `python3 -m pytest -q` (about a second). `tests/test_model.py` is skipped unless `KESTROM_MODEL` is a local Hugging Face
  folder of a chat model whose output layer is not tied to the embedding (we run it with Qwen/Qwen3.5-9B; MLX, Apple Silicon).
- Lint: `ruff check .` (default rules). Do not run `ruff format`: the code keeps its own dense layout.
- Without installing: `PYTHONPATH=verify:edit python3 -m kestrom_verify ...` and `python3 -m kestrom ...`.
- CI (`.github/workflows/test.yml`) runs the tests on Ubuntu with numpy only: MLX and the model tests never run there.

The hooks in `.claude/settings.json` run `ruff check` on every edited `.py` file, and, before a turn ends with `.py` changes in
the tree, the tests and the license boundary.

## Where the facts live

- A module's docstring is its command help (`python -m pydoc kestrom.<module>`; `kestrom-verify` with no arguments). Keep it exact.
- The same facts appear in `README.md` (the tier and tool tables, the results), `docs/guarantees.md` (checks and commands),
  `docs/report-format.md` (report keys), `paper/kestrom.md` and `examples/*/README.md`. A change to a command, a check, a report
  key or a tier changes all of them in the same change. Then run the `doc-sync-reviewer` agent.
- Every number in the README, the paper and the examples comes from a run. Never write or adjust one without the run. When code
  that produced a number changes, say that the number is stale.

## Invariants

- A report has no times in it, and the same inputs give the same bytes (the tests rebuild reports and compare them byte for byte).
- The tiers are not blurred: tier 1 says which outputs stay the same, tier 2 only which weights changed, tier 3 only counts. Never
  state a guarantee that no check computes (`docs/guarantees.md`).
- `mlx`, `mlx_lm` and `transformers` are imported inside functions, so the numpy-only tests and CI run without them.

## The repository

- Public. Issues are welcome; pull requests are not accepted (`CONTRIBUTING.md`). Ask before creating or commenting on issues
  and before pushing.
- The license files are not edited by Claude (deny rules).
