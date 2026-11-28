"""Deterministic container structure and chat-template inspection for GGUF files and safetensors folders.
Detects unreferenced trailing bytes (e.g. appended payloads past the final tensor), inter-tensor gaps or
non-zero padding, overlapping or out-of-bounds tensor spans, and extracts chat_template metadata (length,
SHA-256, full text) for human review. Does not classify natural-language prompt injections or malware families.
Usage: kestrom-verify container MODEL REPORT.json
  MODEL: a folder of safetensors or one GGUF file (any file is read as GGUF, whatever its name)."""
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


def _layout(fh, fname: str, spans, data_start: int, file_size: int, alignment: int) -> list[dict[str, Any]]:
    """The anomalies of one file whose tensors hold the sorted spans [(start, end, name)] of its data section from data_start: a span
    past the end of the file, a span starting before the furthest end so far, and bytes no tensor holds (before the first span,
    between two, after the last) that are not all zero or reach past the next multiple of alignment."""
    anomalies: list[dict[str, Any]] = []
    reached, last = data_start, "__header__"
    for start, end, name in spans:
        if end > file_size:
            anomalies.append({"file": fname, "kind": "out_of_bounds", "tensor": name, "offset": start, "end": end, "file_size": file_size})
        if start < reached:
            anomalies.append({"file": fname, "kind": "overlap", "between": [last, name], "prev_end": reached, "next_start": start})
        elif start > reached:
            expected = _align_up(reached, alignment)
            nz, digest, head_hex = _scan_region(fh, reached, min(start, file_size))
            if start > expected or nz > 0:
                anomalies.append({"file": fname, "kind": "inter_tensor_gap", "between": [last, name], "offset": reached,
                                  "length": start - reached, "expected_offset": expected, "nonzero_bytes": nz, "sha256": digest,
                                  "head_hex": head_hex})
        if end > reached:
            reached, last = end, name
    if file_size > reached:
        expected = _align_up(reached, alignment)
        nz, digest, head_hex = _scan_region(fh, reached, file_size)
        if file_size > expected or nz > 0:
            anomalies.append({"file": fname, "kind": "trailing_bytes", "offset": reached, "length": file_size - reached,
                              "expected_max_end": expected, "nonzero_bytes": nz, "sha256": digest, "head_hex": head_hex})
    return anomalies


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
    anomalies: list[dict[str, Any]] = [] if spans else [{"file": path.name, "kind": "empty_tensors"}]
    with open(path, "rb") as fh:
        anomalies += _layout(fh, path.name, spans, int(reader.data_offset), file_size, alignment)

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
        cfg = json.loads(tc.read_text(encoding="utf-8"))
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
        spans = sorted(
            (data_start + int(v["data_offsets"][0]), data_start + int(v["data_offsets"][1]), k)
            for k, v in hdr.items()
        )
        with open(sf, "rb") as fh:
            found = _layout(fh, sf.name, spans, data_start, sf.stat().st_size, 1)  # safetensors packs its tensors with no padding
        anomalies += found
        if found:
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
    """Inspect a safetensors folder, or a GGUF file (any file, whatever its name, as diff reads it), for container anomalies and
    chat templates."""
    p = Path(path)
    return _inspect_safetensors_folder(p) if p.is_dir() else _inspect_gguf(p)


def document(model) -> dict[str, Any]:
    """The report: the model's files by SHA-256 and what inspect found."""
    fs = weights.files(model)
    return {"model": weights.named(fs, weights.hashes(fs)), "container": inspect(model)}


def main(model, out):
    report.finish(out, document(model))


if __name__ == "__main__":
    main(*sys.argv[1:])
