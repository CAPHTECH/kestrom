#!/usr/bin/env sh
# Zero-download, byte-reproducible verification and acceptance audit for Weight-Space Programming (WSP)
# or declared fine-tuned checkpoints using kestrom-verify (Apache-2.0) and kestrom audit.
# Usage: sh examples/wsp-verify-audit/run.sh [OUT_DIR]
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/out}"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/verify:$ROOT/edit${PYTHONPATH:+:$PYTHONPATH}"

rm -rf "$OUT"
mkdir -p "$OUT"

# 0. Build tiny deterministic safetensors checkpoints (pure Python + NumPy, no downloads)
"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], meta: dict[str, str]) -> None:
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        raw = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = meta
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hdr += b" " * (-len(hdr) % 8)
    path.write_bytes(len(hdr).to_bytes(8, "little") + hdr + b"".join(blobs))


def write_checkpoint(folder: Path, tensors: dict[str, np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_safetensors(folder / "model.safetensors", tensors, {"format": "pt"})
    (folder / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["ToyWSPModel"],
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "tie_word_embeddings": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"}) + "\n",
        encoding="utf-8",
    )


out = Path(sys.argv[1])
rng = np.random.default_rng(0)
base = {
    "model.layers.0.mlp.down_proj.weight": (rng.standard_normal((128, 256)) * 0.05).astype(np.float32),
    "model.layers.1.mlp.down_proj.weight": (rng.standard_normal((128, 256)) * 0.05).astype(np.float32),
    "lm_head.weight": (rng.standard_normal((16, 64)) * 0.05).astype(np.float32),
}

# Declared rank-1 edit touching only rows [0, 4) of layer 0 down_proj
u = rng.standard_normal(4).astype(np.float32)
v = rng.standard_normal(256).astype(np.float32)
delta = (u[:, None] * v[None, :] * 0.005).astype(np.float32)

declared = {k: a.copy() for k, a in base.items()}
declared["model.layers.0.mlp.down_proj.weight"][0:4] += delta

# Undeclared edit: also touches row 10 of layer 1 down_proj outside DECLARED.json
undeclared = {k: a.copy() for k, a in declared.items()}
undeclared["model.layers.1.mlp.down_proj.weight"][10] += np.float32(0.01)

write_checkpoint(out / "base", base)
write_checkpoint(out / "edited_declared", declared)
write_checkpoint(out / "edited_undeclared", undeclared)
(out / "DECLARED.json").write_text(
    json.dumps({"model.layers.0.mlp.down_proj.weight": [[0, 4]]}, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "=== 1. kestrom-verify container (C) ==="
"$PYTHON" -m kestrom_verify container "$OUT/base" "$OUT/container_base.json"
"$PYTHON" -m kestrom_verify container "$OUT/edited_declared" "$OUT/container_declared.json"
"$PYTHON" -m kestrom_verify container "$OUT/edited_undeclared" "$OUT/container_undeclared.json"

echo "=== 2. kestrom-verify diff with DECLARED.json (D) ==="
"$PYTHON" -m kestrom_verify diff "$OUT/base" "$OUT/edited_declared" "$OUT/diff_declared_1.json" "$OUT/DECLARED.json"
"$PYTHON" -m kestrom_verify diff "$OUT/base" "$OUT/edited_declared" "$OUT/diff_declared_2.json" "$OUT/DECLARED.json"
cmp "$OUT/diff_declared_1.json" "$OUT/diff_declared_2.json"
echo "cmp: diff_declared_1.json and diff_declared_2.json are byte-identical"

set +e
"$PYTHON" -m kestrom_verify diff "$OUT/base" "$OUT/edited_undeclared" "$OUT/diff_undeclared.json" "$OUT/DECLARED.json"
rc=$?
set -e
if [ "$rc" -ne 1 ]; then
    echo "expected exit status 1 for undeclared diff, got $rc" >&2
    exit 1
fi
echo "diff undeclared exited 1 as expected (FAIL)"

echo "=== 3. kestrom audit (container + diff + mirror -> JSON / HTML / SARIF) ==="
"$PYTHON" -m kestrom audit "$OUT/edited_declared" "$OUT/audit_declared_1" "$OUT/base" "$OUT/DECLARED.json" nf4 4 64 torch
"$PYTHON" -m kestrom audit "$OUT/edited_declared" "$OUT/audit_declared_2" "$OUT/base" "$OUT/DECLARED.json" nf4 4 64 torch
for f in container.json diff.json mirror.json audit.json audit.html audit.sarif; do
    cmp "$OUT/audit_declared_1/$f" "$OUT/audit_declared_2/$f"
done
echo "cmp: audit_declared_1 and audit_declared_2 artifacts are byte-identical"

set +e
"$PYTHON" -m kestrom audit "$OUT/edited_undeclared" "$OUT/audit_undeclared" "$OUT/base" "$OUT/DECLARED.json" nf4 4 64 torch
rc=$?
set -e
if [ "$rc" -ne 1 ]; then
    echo "expected exit status 1 for undeclared audit, got $rc" >&2
    exit 1
fi
echo "audit undeclared exited 1 as expected (FLAGGED)"
