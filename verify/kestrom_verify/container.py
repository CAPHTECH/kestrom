"""Deterministic container structure and chat-template inspection for GGUF files and safetensors folders.
Detects unreferenced trailing bytes (e.g. appended payloads past the final tensor), inter-tensor gaps or
non-zero padding, overlapping or out-of-bounds tensor spans, and extracts chat_template metadata (length,
SHA-256, full text) for human review. Does not classify natural-language prompt injections or malware families.
Usage: kestrom-verify container MODEL REPORT.json
  MODEL: a folder of safetensors or one GGUF file."""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from . import report, weights

CHUNK = 1 << 20
DEFAULT_GGUF_ALIGNMENT = 32


def _align_up(offset: int, alignment: int) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def _scan_region(fh, start: int, end: int) -> tuple[int, str, str]:
    """Return (nonzero_bytes, sha256_hex, head_hex) over [start, end) without loading it into memory."""
    fh.seek(start)
    remaining = max(0, end - start)
    h = hashlib.sha256()
    nonzero = 0
    head = b""
    while remaining > 0:
        buf = fh.read(min(CHUNK, remaining))
        if not buf:
            break
        if len(head) < 16:
            head += buf[: 16 - len(head)]
        h.update(buf)
        nonzero += len(buf) - buf.count(0)
        remaining -= len(buf)
    return nonzero, h.hexdigest(), head.hex()


def _template_entry(raw: bytes) -> dict[str, Any]:
    return {
        "length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": raw.decode("utf-8", errors="replace"),
    }


def _gguf_alignment(reader) -> int:
    field = reader.fields.get("general.alignment")
    if field is None or not getattr(field, "data", None):
        return DEFAULT_GGUF_ALIGNMENT
    val = int(field.parts[field.data[0]][0])
    return val if val > 0 else DEFAULT_GGUF_ALIGNMENT


def _inspect_gguf(path: Path) -> dict[str, Any]:
    import gguf

    reader = gguf.GGUFReader(path)
    alignment = _gguf_alignment(reader)
    file_size = path.stat().st_size
    spans = sorted((int(t.data_offset), int(t.data_offset + t.n_bytes), t.name) for t in reader.tensors)
    anomalies: list[dict[str, Any]] = []

    if not spans:
        anomalies.append({"file": path.name, "kind": "empty_tensors"})
    else:
        with open(path, "rb") as fh:
            prev_end: int | None = None
            prev_name: str | None = None
            max_end = 0
            for start, end, name in spans:
                if end > file_size:
                    anomalies.append({
                        "file": path.name,
                        "kind": "out_of_bounds",
                        "tensor": name,
                        "offset": start,
                        "end": end,
                        "file_size": file_size,
                    })
                if prev_end is not None:
                    if prev_end > start:
                        anomalies.append({
                            "file": path.name,
                            "kind": "overlap",
                            "between": [prev_name, name],
                            "prev_end": prev_end,
                            "next_start": start,
                        })
                    elif start > prev_end:
                        expected_start = _align_up(prev_end, alignment)
                        nz, digest, head_hex = _scan_region(fh, prev_end, min(start, file_size))
                        if start > expected_start or nz > 0:
                            anomalies.append({
                                "file": path.name,
                                "kind": "inter_tensor_gap",
                                "between": [prev_name, name],
                                "offset": prev_end,
                                "length": start - prev_end,
                                "expected_offset": expected_start,
                                "nonzero_bytes": nz,
                                "sha256": digest,
                                "head_hex": head_hex,
                            })
                if end > max_end:
                    max_end = end
                prev_end, prev_name = end, name

            if file_size > max_end:
                expected_end = _align_up(max_end, alignment)
                nz, digest, head_hex = _scan_region(fh, max_end, file_size)
                if file_size > expected_end or nz > 0:
                    anomalies.append({
                        "file": path.name,
                        "kind": "trailing_bytes",
                        "offset": max_end,
                        "length": file_size - max_end,
                        "expected_max_end": expected_end,
                        "nonzero_bytes": nz,
                        "sha256": digest,
                        "head_hex": head_hex,
                    })

    chat_templates: dict[str, dict[str, Any]] = {}
    for key in sorted(reader.fields):
        if key == "tokenizer.chat_template" or key.startswith("tokenizer.chat_template."):
            field = reader.fields[key]
            if getattr(field, "data", None):
                raw = bytes(field.parts[field.data[0]])
                chat_templates[key] = _template_entry(raw)

    return {
        "format": "gguf",
        "status": "FLAGGED" if anomalies else "PASS",
        "files_scanned": 1,
        "files_clean": 0 if anomalies else 1,
        "alignment": alignment,
        "anomalies": anomalies,
        "chat_templates": chat_templates,
    }


def _extract_folder_chat_templates(folder: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    tc = folder / "tokenizer_config.json"
    if tc.is_file():
        try:
            cfg = json.loads(tc.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cfg = None
        if isinstance(cfg, dict) and cfg.get("chat_template") is not None:
            tpl = cfg["chat_template"]
            text = tpl if isinstance(tpl, str) else json.dumps(tpl, sort_keys=True, ensure_ascii=False)
            out["tokenizer_config.json:chat_template"] = _template_entry(text.encode("utf-8"))
    for name in ("chat_template.jinja", "chat_template.json"):
        p = folder / name
        if p.is_file():
            out[name] = _template_entry(p.read_bytes())
    return out


def _inspect_safetensors_folder(folder: Path) -> dict[str, Any]:
    shards = sorted(folder.glob("*.safetensors"))
    anomalies: list[dict[str, Any]] = []
    dirty_files: set[str] = set()

    for sf in shards:
        hn, hdr = weights.header(sf)
        hdr.pop("__metadata__", None)
        data_start = 8 + hn
        file_size = sf.stat().st_size
        spans = sorted(
            (data_start + int(v["data_offsets"][0]), data_start + int(v["data_offsets"][1]), k)
            for k, v in hdr.items()
        )
        if not spans:
            if file_size > data_start:
                with open(sf, "rb") as fh:
                    nz, digest, head_hex = _scan_region(fh, data_start, file_size)
                anomalies.append({
                    "file": sf.name,
                    "kind": "trailing_bytes",
                    "offset": data_start,
                    "length": file_size - data_start,
                    "nonzero_bytes": nz,
                    "sha256": digest,
                    "head_hex": head_hex,
                })
                dirty_files.add(sf.name)
            continue

        with open(sf, "rb") as fh:
            if spans[0][0] > data_start:
                nz, digest, head_hex = _scan_region(fh, data_start, min(spans[0][0], file_size))
                anomalies.append({
                    "file": sf.name,
                    "kind": "inter_tensor_gap",
                    "between": ["__header__", spans[0][2]],
                    "offset": data_start,
                    "length": spans[0][0] - data_start,
                    "nonzero_bytes": nz,
                    "sha256": digest,
                    "head_hex": head_hex,
                })
                dirty_files.add(sf.name)

            prev_end: int | None = None
            prev_name: str | None = None
            max_end = data_start
            for start, end, name in spans:
                if end > file_size:
                    anomalies.append({
                        "file": sf.name,
                        "kind": "out_of_bounds",
                        "tensor": name,
                        "offset": start,
                        "end": end,
                        "file_size": file_size,
                    })
                    dirty_files.add(sf.name)
                if prev_end is not None:
                    if prev_end > start:
                        anomalies.append({
                            "file": sf.name,
                            "kind": "overlap",
                            "between": [prev_name, name],
                            "prev_end": prev_end,
                            "next_start": start,
                        })
                        dirty_files.add(sf.name)
                    elif start > prev_end:
                        nz, digest, head_hex = _scan_region(fh, prev_end, min(start, file_size))
                        anomalies.append({
                            "file": sf.name,
                            "kind": "inter_tensor_gap",
                            "between": [prev_name, name],
                            "offset": prev_end,
                            "length": start - prev_end,
                            "nonzero_bytes": nz,
                            "sha256": digest,
                            "head_hex": head_hex,
                        })
                        dirty_files.add(sf.name)
                if end > max_end:
                    max_end = end
                prev_end, prev_name = end, name

            if file_size > max_end:
                nz, digest, head_hex = _scan_region(fh, max_end, file_size)
                anomalies.append({
                    "file": sf.name,
                    "kind": "trailing_bytes",
                    "offset": max_end,
                    "length": file_size - max_end,
                    "nonzero_bytes": nz,
                    "sha256": digest,
                    "head_hex": head_hex,
                })
                dirty_files.add(sf.name)

    return {
        "format": "safetensors",
        "status": "FLAGGED" if anomalies else "PASS",
        "files_scanned": len(shards),
        "files_clean": len(shards) - len(dirty_files),
        "anomalies": anomalies,
        "chat_templates": _extract_folder_chat_templates(folder),
    }


def inspect(path) -> dict[str, Any]:
    """Inspect a GGUF file or safetensors folder for container anomalies and chat templates."""
    p = Path(path)
    if p.is_dir():
        return _inspect_safetensors_folder(p)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return _inspect_gguf(p)
    raise SystemExit(f"{path}: expected a safetensors folder or a .gguf file")


def main(model, out):
    res = inspect(model)
    fs = weights.files(model)
    doc = {"model": weights.named(fs, weights.hashes(fs)), "container": res}
    report.finish(out, doc)


if __name__ == "__main__":
    main(*sys.argv[1:])
