# Byte-reproducible verification and CI audit for weight-space edits (tiers 2 and 3)

Weight-Space Programming (WSP), self-rewriting checkpoints, and delivered fine-tunes modify weights directly and claim a bounded
edit scope (`DECLARED.json`). This example verifies those claims with `kestrom-verify` (**Apache-2.0**, free for everyone) and
`kestrom audit` (**PolyForm** / commercial license; see [edit/COMMERCIAL.md](../../edit/COMMERCIAL.md)) using tiny deterministic
safetensors checkpoints generated on the fly—**zero external model downloads**, pure Python + NumPy, running in under two seconds.

```sh
sh examples/wsp-verify-audit/run.sh [OUT_DIR]
```

If `OUT_DIR` already exists and is non-empty, the script refuses to clean it unless the marker file
`OUT_DIR/.kestrom-wsp-verify-audit-out` from a prior run is present.

## What the script runs

0. Generates three deterministic safetensors checkpoints under `OUT_DIR` (default `examples/wsp-verify-audit/out`):
   - `base/`: baseline checkpoint (`model.safetensors`, `config.json`, `tokenizer_config.json`).
   - `edited_declared/`: rank-1 update confined to rows `[0, 4)` of `model.layers.0.mlp.down_proj.weight`, matching `DECLARED.json`.
   - `edited_undeclared/`: the same declared edit plus an undeclared change to row `10` of `model.layers.1.mlp.down_proj.weight`.
1. **`kestrom-verify container`** (Apache-2.0, tier 2, check `C`): verifies zero trailing bytes, no layout anomalies, and extracts
   `chat_template` metadata on all three checkpoints (`PASS`).
2. **`kestrom-verify diff`** (Apache-2.0, tier 2, check `D`): runs twice on `edited_declared` against `base` and `DECLARED.json`
   (`D the changes inside the declaration: 1 of 1 PASS`), checks with `cmp` that both JSON reports are byte-identical, and
   verifies `edited_undeclared` exits `1` (`FAIL`, naming `model.layers.1.mlp.down_proj.weight` outside the declaration).
3. **`kestrom audit`** (PolyForm / commercial): runs `container` + `diff` + `inspect mirror` (`nf4` 4-bit group 64, pure NumPy)
   twice on `edited_declared` (`PASS`, exiting `0`), verifies with `cmp` that `container.json`, `diff.json`, `mirror.json`,
   `audit.json`, `audit.html`, and `audit.sarif` rebuild byte for byte, and runs on `edited_undeclared` (`FLAGGED`, exiting `1`,
   emitting a `KESTROM-DIFF-UNDECLARED` SARIF finding).

## Scope of the checks

- Tier 2 (`container`, `diff` + `D`) checks container structure and that every changed tensor, row, metadata field, and file
  lies inside `DECLARED.json`. Changing an internal layer can still move outputs at any position; `diff` does not bound outputs.
- Tier 3 (`inspect mirror` inside `audit`) checks for quantization-conditioned rounding anomalies; zero findings is statistical
  evidence for the scanned quantizer, not a proof that no backdoor exists ([docs/guarantees.md](../../docs/guarantees.md)).
- `audit.json` records the exact input paths (`model.path` and `base.path`), so byte-for-byte `cmp` reproduction holds when
  rebuilt on the same inputs at the same paths (for example, the same relative layout).

## GitHub Actions CI gates (WSP benchmarks & model delivery)

### 1. Standalone Apache-2.0 Verifier Gate (`kestrom-verify` only)

Free for everyone—including model receivers, auditors, and public WSP benchmarks (such as IWIBench)—with no `kestrom` / PolyForm
dependency. PyPI `kestrom-verify 0.1.0` already includes `diff`, `range`, and `compare`; `container` is in `main` (`>=0.2.0`):

```yaml
      - name: Install kestrom-verify (Apache-2.0)
        # In production CI, pin a release tag (e.g. @verify-v0.2.0) or commit SHA instead of @main.
        run: pip install "git+https://github.com/CAPHTECH/kestrom.git@main#subdirectory=verify"

      - name: Verify container integrity and declared weight-space scope
        run: |
          kestrom-verify container "$DELIVERED_DIR" container.json
          kestrom-verify diff "$BASE_DIR" "$DELIVERED_DIR" diff.json "$DECLARED_JSON"
```

### 2. Full Audit + SARIF Gate (`kestrom audit`)

Adds `kestrom audit` (`container` + `diff` + `inspect mirror`, emitting `audit.json`, `audit.html`, and `audit.sarif`). Installing
`kestrom` (`subdirectory=edit`) is governed by PolyForm (free for companies with fewer than 100 people, noncommercial use, or an
evaluation trial of less than 32 consecutive calendar days; commercial and client-engagement agreements are in
[edit/COMMERCIAL.md](../../edit/COMMERCIAL.md)).

Run `kestrom audit` directly (it already runs `container` and `diff` and writes `audit.sarif` even when exiting `1` on `FLAGGED`),
capture its exit code, upload the SARIF file with `if: always()`, and fail the job afterward:

```yaml
    permissions:
      contents: read
      security-events: write

    steps:
      - name: Install kestrom-verify and kestrom
        # In production CI, pin a release tag or commit SHA instead of @main.
        run: |
          pip install \
            "git+https://github.com/CAPHTECH/kestrom.git@main#subdirectory=verify" \
            "git+https://github.com/CAPHTECH/kestrom.git@main#subdirectory=edit"

      - name: Run acceptance audit (container + diff + mirror -> SARIF)
        id: audit
        run: |
          rc=0
          kestrom audit "$DELIVERED_DIR" audit-out "$BASE_DIR" "$DECLARED_JSON" nf4 4 64 torch || rc=$?
          echo "rc=$rc" >> "$GITHUB_OUTPUT"

      - name: Upload Kestrom SARIF to GitHub Code Scanning
        if: always()
        uses: github/codeql-action/upload-sarif@v4
        with:
          sarif_file: audit-out/audit.sarif

      - name: Enforce audit verdict
        run: |
          if [ "${{ steps.audit.outputs.rc }}" -ne 0 ]; then
            exit "${{ steps.audit.outputs.rc }}"
          fi
```
