"""Tests for kestrom audit (numpy only): deterministic byte-identical JSON/HTML/SARIF, diff declaration failures, and mirror flags."""
import json

import numpy as np
import pytest
from test_verify import folder

from kestrom import audit as ka


def test_audit_standalone_no_findings_and_byte_reproducible(tmp_path):
    rng = np.random.default_rng(0)
    w = {
        "layers.0.mlp.down_proj.weight": (rng.standard_normal((128, 256)) * 0.05).astype(np.float32),
        "lm_head.weight": (rng.standard_normal((16, 64)) * 0.05).astype(np.float32),
    }
    model = folder(tmp_path / "model", w)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"

    ka.audit(str(model), str(out1), mode="nf4")
    ka.audit(str(model), str(out2), mode="nf4")

    for name in ("audit.json", "audit.html", "audit.sarif"):
        assert (out1 / name).is_file()
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()

    doc = json.loads((out1 / "audit.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "kestrom-audit-v1"
    assert doc["verdict"] == "NO_FINDINGS"
    assert [c["id"] for c in doc["checks"]] == ["mirror"]
    assert doc["checks"][0]["status"] == "NO_FINDINGS"
    assert "VERDICT: NO_FINDINGS" in (out1 / "audit.html").read_text(encoding="utf-8")
    sarif = json.loads((out1 / "audit.sarif").read_text(encoding="utf-8"))
    assert sarif["runs"][0]["results"] == []


def test_audit_with_base_and_matching_declaration_passes(tmp_path):
    rng = np.random.default_rng(1)
    w_base = {
        "layers.0.mlp.down_proj.weight": (rng.standard_normal((128, 256)) * 0.05).astype(np.float32),
        "layers.1.mlp.down_proj.weight": (rng.standard_normal((128, 256)) * 0.05).astype(np.float32),
    }
    w_edit = dict(w_base)
    w_edit["layers.0.mlp.down_proj.weight"] = w_base["layers.0.mlp.down_proj.weight"] + 0.001

    base = folder(tmp_path / "base", w_base)
    edit = folder(tmp_path / "edit", w_edit)
    decl = tmp_path / "declared.json"
    decl.write_text(json.dumps({"layers.0.mlp.down_proj.weight": "all"}), encoding="utf-8")

    out_dir = tmp_path / "out_pass"
    ka.audit(str(edit), str(out_dir), str(base), str(decl), mode="nf4")

    doc = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "PASS"
    assert doc["checks"][0]["id"] == "diff"
    assert doc["checks"][0]["tier"] == 2
    assert doc["checks"][0]["status"] == "PASS"
    assert doc["checks"][1]["id"] == "mirror"
    assert doc["checks"][1]["status"] == "NO_FINDINGS"
    assert "VERDICT: PASS" in (out_dir / "audit.html").read_text(encoding="utf-8")
    sarif = json.loads((out_dir / "audit.sarif").read_text(encoding="utf-8"))
    assert sarif["runs"][0]["results"] == []


def test_audit_with_base_and_declared_flags_undeclared_change(tmp_path):
    rng = np.random.default_rng(1)
    w_base = {
        "layers.0.mlp.down_proj.weight": (rng.standard_normal((128, 256)) * 0.05).astype(np.float32),
        "layers.1.mlp.down_proj.weight": (rng.standard_normal((128, 256)) * 0.05).astype(np.float32),
    }
    w_edit = dict(w_base)
    w_edit["layers.1.mlp.down_proj.weight"] = w_base["layers.1.mlp.down_proj.weight"] + 0.01

    base = folder(tmp_path / "base", w_base)
    edit = folder(tmp_path / "edit", w_edit)
    decl = tmp_path / "declared.json"
    decl.write_text(json.dumps({"layers.0.mlp.down_proj.weight": "all"}), encoding="utf-8")

    out_dir = tmp_path / "out_fail"
    with pytest.raises(SystemExit) as exc:
        ka.audit(str(edit), str(out_dir), str(base), str(decl), mode="nf4")
    assert exc.value.code == 1

    doc = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "FLAGGED"
    assert doc["checks"][0]["id"] == "diff"
    assert doc["checks"][0]["tier"] == 2
    assert doc["checks"][0]["status"] == "FLAGGED"
    assert "layers.1.mlp.down_proj.weight" in doc["checks"][0]["outside"]
    sarif = json.loads((out_dir / "audit.sarif").read_text(encoding="utf-8"))
    assert sarif["runs"][0]["results"][0]["ruleId"] == "KESTROM-DIFF-UNDECLARED"


def test_audit_refuses_out_inside_model_and_an_empty_scan(tmp_path):
    model = folder(tmp_path / "model", {"lm_head.weight": np.zeros((16, 64), np.float32)})
    with pytest.raises(SystemExit, match="inside"):
        ka.audit(str(model), str(model / "audit"), mode="nf4")
    assert not (model / "audit").exists()
    with pytest.raises(SystemExit, match="scanned no tensor"):
        ka.audit(str(tmp_path / "empty"), str(tmp_path / "out"), mode="nf4")


def test_audit_flags_mirror_anomaly_with_sarif(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    model = folder(tmp_path / "model", {"layers.0.mlp.down_proj.weight": rng.standard_normal((128, 256)).astype(np.float32)})

    def fake_mirror(model_dir, out, prompts=None, bits=None, group=None, mode="affine", runtime=None):
        payload = {
            "model": str(model_dir),
            "quantizer": {"mode": mode, "bits": 4, "group": 64, "runtime": "torch"},
            "tensors_scanned": 1,
            "flagged": ["layers.0.mlp.down_proj.weight"],
            "reasons": {"layers.0.mlp.down_proj.weight": ["rho1=9.50"]},
            "on_grid": [],
            "weight_summary": {"max_rho1": 9.5, "max_b85": 0.72, "max_collapse_ge90": 5},
            "tensors": {},
        }
        (tmp_path / "out_mirror").mkdir(exist_ok=True)
        open(out, "w", encoding="utf-8").write(json.dumps(payload))

    monkeypatch.setattr(ka.mirror, "mirror", fake_mirror)
    out_dir = tmp_path / "out_mirror"
    with pytest.raises(SystemExit) as exc:
        ka.audit(str(model), str(out_dir))
    assert exc.value.code == 1

    sarif = json.loads((out_dir / "audit.sarif").read_text(encoding="utf-8"))
    assert sarif["runs"][0]["results"][0]["ruleId"] == "KESTROM-MIRROR-FLAGGED"
    assert "VERDICT: FLAGGED" in (out_dir / "audit.html").read_text(encoding="utf-8")
