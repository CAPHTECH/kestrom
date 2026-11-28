# Byte-reproducible verification and CI audit for weight-space edits (tiers 2 and 3)

Weight-Space Programming (WSP), self-rewriting checkpoints, and delivered fine-tunes modify weights directly and claim a bounded
edit scope (`DECLARED.json`). This example verifies those claims with `kestrom-verify` (Apache-2.0) and `kestrom audit` using
tiny deterministic safetensors checkpoints generated on the fly—**zero external model downloads**, pure Python + NumPy, running
in under two seconds.

```sh
sh examples/wsp-verify-audit/run.sh [OUT_DIR]
```

## What the script runs

0. Generates three deterministic safetensors checkpoints under `OUT_DIR` (default `examples/wsp-verify-audit/out`):
   - `base/`: baseline checkpoint (`model.safetensors`, `config.json`, `tokenizer_config.json`).
   - `edited_declared/`: rank-1 update confined to rows `[0, 4)` of `model.layers.0.mlp.down_proj.weight`, matching `DECLARED.json`.
   - `edited_undeclared/`: the same declared edit plus an undeclared change to row `10` of `model.layers.1.mlp.down_proj.weight`.
1. **`kestrom-verify container`** (tier 2, check `C`): verifies zero trailing bytes, no layout anomalies, and extracts
   `chat_template` metadata on all three checkpoints (`PASS`).
2. **`kestrom-verify diff`** (tier 2, check `D`): runs twice on `edited_declared` against `base` and `DECLARED.json`
   (`D: 1 of 1 PASS`), checks with `cmp` that both JSON reports are byte-identical, and verifies `edited_undeclared` exits `1`
   (`FAIL`, naming `model.layers.1.mlp.down_proj.weight` outside the declaration).
3. **`kestrom audit`**: runs `container` + `diff` + `inspect mirror` (`nf4` 4-bit group 64, pure NumPy) twice on
   `edited_declared` (`PASS`, exiting `0`), verifies with `cmp` that `container.json`, `diff.json`, `mirror.json`, `audit.json`,
   `audit.html`, and `audit.sarif` rebuild byte for byte, and runs on `edited_undeclared` (`FLAGGED`, exiting `1`, emitting a
   `KESTROM-DIFF-UNDECLARED` SARIF finding).

## Scope of the checks

- Tier 2 (`container`, `diff` + `D`) checks container structure and that every changed tensor, row, metadata field, and file
  lies inside `DECLARED.json`. Changing an internal layer can still move outputs at any position; `diff` does not bound outputs.
- Tier 3 (`inspect mirror` inside `audit`) checks for quantization-conditioned rounding anomalies; zero findings is statistical
  evidence for the scanned quantizer, not a proof that no backdoor exists (`docs/guarantees.md`).

## GitHub Actions CI gate (WSP benchmarks & model delivery)

For WSP evaluation harnesses (such as IWIBench) or model delivery pipelines, run `kestrom-verify` / `kestrom audit` in CI and
upload `audit.sarif` to GitHub Code Scanning (`if: always()` ensures findings still appear when `kestrom audit` exits `1` on
`FLAGGED`):

```yaml
      - name: Install Kestrom verification and audit tools
        # kestrom-verify (Apache-2.0) is on PyPI; kestrom audit installs from this repository
        # (from a local checkout of CAPHTECH/kestrom, use: pip install ./verify ./edit)
        run: pip install kestrom-verify "git+https://github.com/CAPHTECH/kestrom.git#subdirectory=edit"

      - name: Verify declared weight-space scope and generate SARIF
        run: |
          kestrom-verify container "$DELIVERED_DIR" container.json
          kestrom-verify diff "$BASE_DIR" "$DELIVERED_DIR" diff.json "$DECLARED_JSON"
          kestrom audit "$DELIVERED_DIR" audit-out "$BASE_DIR" "$DECLARED_JSON" nf4 4 64 torch

      - name: Upload Kestrom SARIF to GitHub Code Scanning
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: audit-out/audit.sarif
```
