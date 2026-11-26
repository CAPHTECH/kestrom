"""One-command acceptance audit: weight diff (optional BASE_DIR + DECLARED.json) and anti-quantization mirror scan, producing deterministic audit.json, audit.html, and audit.sarif (no timestamps; rebuilds byte for byte).
  kestrom audit MODEL_DIR OUT_DIR [BASE_DIR|none] [DECLARED.json|none] [MODE] [BITS] [GROUP] [RUNTIME]
      writes diff.json (when BASE_DIR is given), mirror.json, audit.json, audit.html, and audit.sarif into OUT_DIR.
      Exits 0 on PASS (BASE_DIR + DECLARED.json checked with no outside changes and no mirror flags) or NO_FINDINGS (no declaration checked and no mirror flags), 1 on FLAGGED.
      Refuses an OUT_DIR inside MODEL_DIR or BASE_DIR, and a model of which mirror scans no tensor."""
import html
import json
from pathlib import Path
from typing import Any

from kestrom_verify import diff as vdiff
from kestrom_verify import weights

from . import mirror


def _opt(val):
    return None if val in (None, "", "none") else val


def _artifact_files(folder: Path) -> dict[str, str]:
    fs = weights.files(folder)
    return weights.named(fs, weights.hashes(fs))


def _render_sarif(doc: dict[str, Any]) -> dict[str, Any]:
    rules_meta = [
        {
            "id": "KESTROM-DIFF-UNDECLARED",
            "name": "UndeclaredWeightOrFileChange",
            "shortDescription": {"text": "Delivered model contains tensor or file changes outside DECLARED.json."},
            "defaultConfiguration": {"level": "error"},
        },
        {
            "id": "KESTROM-MIRROR-FLAGGED",
            "name": "QuantizationMirrorAnomaly",
            "shortDescription": {"text": "Self-contrast quantization mirror scan flagged anomalous weight patterns."},
            "defaultConfiguration": {"level": "error"},
        },
    ]
    results = []
    for chk in doc["checks"]:
        if chk["id"] == "diff" and chk["status"] == "FLAGGED":
            for item in chk["outside"]:
                results.append({
                    "ruleId": "KESTROM-DIFF-UNDECLARED",
                    "level": "error",
                    "message": {"text": f"Undeclared change outside declaration: {item}"},
                    "locations": [{"physicalLocation": {"artifactLocation": {"uri": str(doc["model"]["path"])}}}],
                })
        elif chk["id"] == "mirror" and chk["status"] == "FLAGGED":
            for tensor_name in chk["flagged"]:
                reasons = chk["reasons"].get(tensor_name, [])
                results.append({
                    "ruleId": "KESTROM-MIRROR-FLAGGED",
                    "level": "error",
                    "message": {"text": f"Flagged tensor {tensor_name}: {', '.join(reasons) or 'mirror anomaly'}"},
                    "locations": [{"physicalLocation": {"artifactLocation": {"uri": str(doc["model"]["path"])}}}],
                })
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "kestrom-audit", "rules": rules_meta}},
            "results": results,
        }],
    }


def _render_html(doc: dict[str, Any]) -> str:
    esc = html.escape
    verdict = doc["verdict"]
    badge_cls = "badge-pass" if verdict == "PASS" else ("badge-neutral" if verdict == "NO_FINDINGS" else "badge-fail")
    rows = []
    for chk in doc["checks"]:
        st = chk["status"]
        st_cls = "badge-pass" if st == "PASS" else ("badge-neutral" if st == "NO_FINDINGS" else ("badge-warn" if st == "MEASURED" else "badge-fail"))
        if chk["id"] == "diff":
            s = chk["summary"]
            detail = (
                f"tensors changed {s['tensors changed']} of {s['tensors']}, "
                f"other files changed {s['other files changed']}"
            )
            if chk["outside"]:
                detail += f"; outside declaration: {', '.join(chk['outside'][:10])}"
        else:
            ws = chk["weight_summary"]
            detail = (
                f"scanned {chk['tensors_scanned']} tensors ({chk['quantizer']['mode']} "
                f"{chk['quantizer']['bits']}-bit g{chk['quantizer']['group']} {chk['quantizer']['runtime']}); "
                f"flagged {len(chk['flagged'])}; max_rho1={ws.get('max_rho1', 0.0):.4f}, "
                f"max_b85={ws.get('max_b85', 0.0):.4f}, max_ge90={ws.get('max_collapse_ge90', 0)}"
            )
        rows.append(
            f"<tr><td><strong>{esc(chk['id'])}</strong></td>"
            f"<td>Tier {chk['tier']}</td>"
            f"<td><span class=\"badge {st_cls}\">{esc(st)}</span></td>"
            f"<td>{esc(detail)}</td></tr>"
        )
    file_rows = "".join(
        f"<tr><td class=\"mono\">{esc(k)}</td><td class=\"mono\">{esc(v)}</td></tr>"
        for k, v in sorted(doc["model"]["files"].items())
    )
    repro = "\n".join(doc["reproduce"])
    lims = "".join(f"<li>{esc(x)}</li>" for x in doc["limitations"])
    base_txt = esc(doc["base"]["path"]) if doc["base"] else "(none — standalone scan)"
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\" />\n"
        "<title>Kestrom Acceptance Audit Report</title>\n<style>\n"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f6f8fa;color:#1f2328;margin:0;padding:24px;font-size:14px;line-height:1.5}\n"
        ".page{max-width:960px;margin:0 auto;background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:28px 32px}\n"
        "header{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2px solid #1f2328;padding-bottom:14px;margin-bottom:20px}\n"
        "h1{margin:0 0 4px;font-size:20px}h2{font-size:15px;margin:22px 0 8px;padding-left:10px;border-left:4px solid #0969da}\n"
        ".badge{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700;font-size:12px}\n"
        ".badge-pass{background:#dafbe1;color:#116329}.badge-neutral{background:#eaeef2;color:#24292f}.badge-warn{background:#fff8c5;color:#7d4e00}.badge-fail{background:#ffebe9;color:#cf222e}\n"
        "table{width:100%;border-collapse:collapse;margin:8px 0 14px;font-size:13px}th,td{border:1px solid #d0d7de;padding:7px 9px;text-align:left;vertical-align:top}\n"
        "th{background:#f6f8fa}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;word-break:break-all}\n"
        "pre{background:#161b22;color:#e6edf3;padding:12px 14px;border-radius:8px;overflow-x:auto;font-size:12px}\n"
        "@media print{body{background:#fff;padding:0}.page{border:none}}\n"
        "</style>\n</head>\n<body>\n<div class=\"page\">\n"
        f"<header><div><h1>Kestrom Acceptance Audit Report</h1><div>Model: <span class=\"mono\">{esc(doc['model']['path'])}</span><br/>Base: <span class=\"mono\">{base_txt}</span></div></div>"
        f"<div><span class=\"badge {badge_cls}\">VERDICT: {esc(verdict)}</span></div></header>\n"
        "<h2>1. Check Summary</h2>\n"
        "<table><thead><tr><th>Check</th><th>Tier</th><th>Status</th><th>Details</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>\n"
        "<h2>2. Model File SHA-256 Manifest</h2>\n"
        f"<table><thead><tr><th>File</th><th>SHA-256</th></tr></thead><tbody>{file_rows}</tbody></table>\n"
        "<h2>3. Reproduction Commands</h2>\n"
        f"<pre>{esc(repro)}</pre>\n"
        "<h2>4. Scope &amp; Limitations</h2>\n"
        f"<ul>{lims}</ul>\n"
        "</div>\n</body>\n</html>\n"
    )


def audit(
    model_dir,
    out_dir,
    base_dir=None,
    declared_file=None,
    mode=mirror.DEFAULT_MODE,
    bits=None,
    group=None,
    runtime=None,
):
    """Run integrated acceptance audit (optional diff + mirror scan) and write audit.json, audit.html, and audit.sarif."""
    base = _opt(base_dir)
    decl_path = _opt(declared_file)
    if decl_path is not None and base is None:
        raise SystemExit("DECLARED.json requires BASE_DIR")

    out_path = Path(out_dir)
    for d in (model_dir, base):
        if d is not None and out_path.resolve().is_relative_to(Path(d).resolve()):
            raise SystemExit(f"OUT_DIR {out_dir} is inside {d}: the audit would read its own reports as model files")
    out_path.mkdir(parents=True, exist_ok=True)

    checks = []
    reproduce = []
    diff_status = None
    model_files = base_files = None
    if base is not None:
        diff_res = vdiff.declared_diff(base, model_dir, decl_path)
        model_files, base_files = diff_res["edited"], diff_res["original"]
        (out_path / "diff.json").write_text(json.dumps(diff_res, ensure_ascii=False, indent=1), encoding="utf-8")
        outside = diff_res.get("declared", {}).get("outside", [])
        diff_status = ("FLAGGED" if outside else "PASS") if decl_path is not None else "MEASURED"
        checks.append({
            "id": "diff",
            "tier": 2,
            "status": diff_status,
            "summary": diff_res["diff"],
            "outside": outside,
        })
        decl_arg = f" {decl_path}" if decl_path is not None else ""
        reproduce.append(f"kestrom-verify diff {base} {model_dir} diff.json{decl_arg}")

    mirror_out = out_path / "mirror.json"
    mirror.mirror(model_dir, str(mirror_out), None, bits, group, mode, runtime)
    mres = json.loads(mirror_out.read_text(encoding="utf-8"))
    if not mres["tensors_scanned"]:
        raise SystemExit(f"mirror scanned no tensor of {model_dir} (no .safetensors in the folder, or no tensor divides by GROUP)")
    mirror_status = "FLAGGED" if mres.get("flagged") else "NO_FINDINGS"
    checks.append({
        "id": "mirror",
        "tier": 3,
        "status": mirror_status,
        "quantizer": mres["quantizer"],
        "tensors_scanned": mres["tensors_scanned"],
        "flagged": mres["flagged"],
        "reasons": mres["reasons"],
        "weight_summary": mres["weight_summary"],
    })
    q = mres["quantizer"]
    reproduce.append(
        f"kestrom inspect mirror {model_dir} mirror.json none {q['bits']} {q['group']} {q['mode']} {q['runtime']}"
    )

    if diff_status == "FLAGGED" or mirror_status == "FLAGGED":
        verdict = "FLAGGED"
    elif diff_status == "PASS":
        verdict = "PASS"
    else:
        verdict = "NO_FINDINGS"
    doc: dict[str, Any] = {
        "schema": "kestrom-audit-v1",
        "verdict": verdict,
        "model": {"path": str(model_dir), "files": model_files or _artifact_files(Path(model_dir))},
        "base": {"path": str(base), "files": base_files} if base is not None else None,
        "checks": checks,
        "reproduce": reproduce,
        "limitations": [
            "Tier 2 diff checks weight changes against DECLARED.json; it does not bound model outputs.",
            "Tier 3 inspect mirror provides statistical forensic evidence for the scanned quantizer; zero flagged tensors is not a mathematical proof of backdoor absence.",
        ],
    }

    (out_path / "audit.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_path / "audit.html").write_text(_render_html(doc), encoding="utf-8")
    (out_path / "audit.sarif").write_text(json.dumps(_render_sarif(doc), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(
        f"audit verdict={verdict} | diff={diff_status or 'skipped'} | mirror={mirror_status} "
        f"({len(mres['flagged'])} flagged of {mres['tensors_scanned']}) | out={out_path}",
        flush=True,
    )
    if verdict == "FLAGGED":
        raise SystemExit(1)
