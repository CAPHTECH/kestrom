"""Fast end-to-end test for examples/wsp-verify-audit/run.sh (pure NumPy, zero downloads)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "wsp-verify-audit" / "run.sh"


def test_wsp_verify_audit_example_runs_and_is_byte_reproducible(tmp_path):
    out_dir = tmp_path / "wsp_out"
    env = dict(os.environ, PYTHON=sys.executable)
    res = subprocess.run(
        ["sh", str(SCRIPT), str(out_dir)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert (out_dir / ".kestrom-wsp-verify-audit-out").is_file()
    assert "D the changes inside the declaration: 1 of 1 PASS" in res.stdout
    assert "audit verdict=PASS" in res.stdout
    assert "audit verdict=FLAGGED" in res.stdout

    assert (out_dir / "diff_declared_1.json").read_bytes() == (out_dir / "diff_declared_2.json").read_bytes()
    diff_bad = json.loads((out_dir / "diff_undeclared.json").read_text(encoding="utf-8"))
    assert diff_bad["declared"]["outside"] == ["model.layers.1.mlp.down_proj.weight"]

    for name in ("container.json", "diff.json", "mirror.json", "audit.json", "audit.html", "audit.sarif"):
        a = (out_dir / "audit_declared_1" / name).read_bytes()
        b = (out_dir / "audit_declared_2" / name).read_bytes()
        assert a == b, f"{name} differed across identical rebuilds"

    audit_pass = json.loads((out_dir / "audit_declared_1" / "audit.json").read_text(encoding="utf-8"))
    assert audit_pass["verdict"] == "PASS"
    assert [c["status"] for c in audit_pass["checks"]] == ["PASS", "PASS", "NO_FINDINGS"]

    audit_flag = json.loads((out_dir / "audit_undeclared" / "audit.json").read_text(encoding="utf-8"))
    assert audit_flag["verdict"] == "FLAGGED"
    sarif_flag = json.loads((out_dir / "audit_undeclared" / "audit.sarif").read_text(encoding="utf-8"))
    assert sarif_flag["runs"][0]["results"][0]["ruleId"] == "KESTROM-DIFF-UNDECLARED"

    # Re-running against a marked OUT_DIR cleans and succeeds cleanly
    res2 = subprocess.run(
        ["sh", str(SCRIPT), str(out_dir)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode == 0, f"rerun failed:\nstdout:\n{res2.stdout}\nstderr:\n{res2.stderr}"


def test_wsp_verify_audit_refuses_nonempty_unmarked_dir(tmp_path):
    victim = tmp_path / "existing"
    victim.mkdir()
    sentinel = victim / "important.txt"
    sentinel.write_text("keep me\n", encoding="utf-8")

    env = dict(os.environ, PYTHON=sys.executable)
    res = subprocess.run(
        ["sh", str(SCRIPT), str(victim)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2
    assert "refusing to clean" in res.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
