"""The shared part of the guarantee, pure Python: the hashes, the report file (JSON, no times: the same inputs give the same bytes),
the verdict read back from a report, its summary in markdown, and the greedy comparison of a row edit."""
import hashlib
import json
from pathlib import Path

MAX_TOKENS = 512  # greedy decoding, both runtimes


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(open(path, encoding="utf-8").read())


def lines(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def output_row(prompt, finish, tokens, text):
    """A greedy output as generate writes it, every runtime: the keys of the prompt but ids, finish (stop or length), the tokens
    without the stop token, their text."""
    return dict({k: v for k, v in prompt.items() if k != "ids"}, finish=finish, tokens=tokens, text=text)


def write_lines(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)


def check(name, n, of, empty_passes=False):
    """(name, count, of, passed): passed when count equals of; a check over no sequences or prompts fails (it checked nothing),
    except D over no changes."""
    return name, n, of, n == of and (of > 0 or empty_passes)


def weight_checks(r):
    w = r["weights"]
    return [check("W the files equal but for the changed rows of the output layer", w["files equal but for the changed rows"], w["files"])]


TIED = " (those with no changed id: the output layer is tied to the embedding)"  # G1 and G2 when the changed rows are also inputs


def forced_checks(r):
    f, out = r["forced"], r["forced"].get("sequences_outside")
    return [check("U0 teacher forcing, the original equal to itself", f["original_equal"], f["sequences"]),
            check("G1 teacher forcing, every logit of an unchanged row equal" + (TIED if out is not None else ""), f["other_equal"],
                  f["sequences"] - (out or 0))]


def greedy_checks(r):
    g, out = r["greedy"], r["greedy"].get("prompts with a changed id")
    return [check("U0 greedy, the original twice the same", g["original twice the same"], g["prompts"]),
            check("G2 greedy, the same output or the first difference in the range" + (TIED if out is not None else ""),
                  g["same or first difference in the range"], g["prompts"] - (out or 0))]


def measure_checks(r):
    """No guarantee: the only check is that the original is equal to itself; the rest is how much changed."""
    m = r["measure"]
    out = [check("U0 teacher forcing, the original equal to itself", m["original_equal"], m["sequences"])]
    if "measure greedy" in r:
        out.append(check("U0 greedy, the original twice the same", r["measure greedy"]["original twice the same"], r["measure greedy"]["prompts"]))
    return out


def declared_checks(r):
    d = r["declared"]
    return [check("D the changes inside the declaration", d["inside"], d["changes"], empty_passes=True)]


CHECKS = {"weights": weight_checks, "declared": declared_checks, "forced": forced_checks, "greedy": greedy_checks, "measure": measure_checks}  # by a key of the report
NO_GUARANTEE = "no guarantee: the range is the whole model (measured only)"


def verdict(r):
    """[(check, count, of, passed)] for every part the report has."""
    return [c for key, f in CHECKS.items() if key in r for c in f(r)]


def finish(path, r):
    """Write the report, print each check, and exit with status 1 if one fails (the report is written either way)."""
    open(path, "w", encoding="utf-8").write(json.dumps(r, ensure_ascii=False, indent=1))
    v = verdict(r)
    for name, n, of, ok in v:
        print(f"{name}: {n} of {of}", "PASS" if ok else "FAIL", flush=True)
    if "measure" in r:
        print(NO_GUARANTEE, flush=True)
    if not all(ok for *_, ok in v):
        raise SystemExit(1)


def summary(path):
    """A report in markdown: the verdict, the totals, and the hashes of what it was made from."""
    r = read(path)
    out = [f"# Verification report: {path}", "", "| Verdict | Check | Count |", "|---|---|---:|"]
    out += [f"| {'PASS' if ok else 'FAIL'} | {name} | {n} / {of} |" for name, n, of, ok in verdict(r)]
    if "measure" in r:
        out += ["", f"**{NO_GUARANTEE}**"]
    parts = ("forced", "greedy", "weights", "measure", "measure greedy", "diff", "declared")
    for key in parts:
        if key in r:
            out += ["", f"## {key}", ""] + [f"- {k}: {len(v) if isinstance(v, list) else v}" for k, v in r[key].items()]
    out += ["", "## Inputs", ""]
    for key, v in r.items():
        if key not in parts + ("rows", "tensors"):
            out += [f"- {key} / {k}: {x}" for k, x in v.items()] if isinstance(v, dict) else [f"- {key}: {v}"]
    return "\n".join(out) + "\n"


def first_difference(x, y):
    return next((i for i, (s, t) in enumerate(zip(x, y)) if s != t), min(len(x), len(y)))


STOP = None  # the end of an output that stopped (finish "stop"), compared like a token


def outputs(files):
    """The greedy outputs (JSONL of id, finish and tokens) of the original, the original again and the edited, each as (id, the tokens
    and STOP if it stopped, the row), in the same order of ids."""
    runs = [[(x["id"], x["tokens"] + ([STOP] if x["finish"] == "stop" else []), x) for x in lines(f)] for f in files]
    assert [i for i, _, _ in runs[0]] == [i for i, _, _ in runs[1]] == [i for i, _, _ in runs[2]], "the outputs are not of the same prompts"
    return zip(*runs)


def unchanged(files):
    """files: greedy outputs of the original, the original again, the edited. The counts of an edit with no range."""
    rows = list(outputs(files))
    return {"prompts": len(rows), "original twice the same": sum(s == s2 for (_, s, _), (_, s2, _), _ in rows),
            "same": sum(s == t for (_, s, _), _, (_, t, _) in rows)}


def greedy(range_file, files, decode, out, prompts=None):
    """files: the greedy outputs (JSONL of id, finish and tokens) of the original, of the original again, of the edited weights or
    decoding; an output that stopped ends with STOP, so a stop against more tokens is a difference. The range is the changed rows of
    the report range_file (a range report whose every check passed): per prompt, the original twice the same (U0), and the edited
    output the same or first different where the original's or the edited token's row changed (G2: before that, every logit of an
    unchanged row is equal, so the top token can differ only there). With range_file "events", the range is the events the edited
    output records (its key "events": the output positions where a backtrack or a fired condition first acted): G2, the same output
    or the first difference at or after the first event (before it, the edited decoding is the original's). The events are what the
    edited decoding says; G2 checks the output against them, not that a loop or a condition was really there. A range whose output
    layer is tied to the embedding needs prompts, the PROMPTS.json the outputs were made from, and only it: a prompt holding a
    changed id is computed with changed inputs from its start, so it is counted outside the check, not checked; in an output, the
    range starts at the first changed id either model writes, the same in both or not (the edited model then reads a changed input)."""
    events = range_file == "events"
    if not events:
        r = read(range_file)
        if "weights" not in r or not all(ok for *_, ok in verdict(r)):
            raise SystemExit(f"{range_file} is not a range report whose every check passed")
    changed = set() if events else set(r["weights"]["changed rows"])
    tied = not events and r["weights"].get("tied to the embedding", False)
    if tied != (prompts is not None):
        raise SystemExit("PROMPTS.json goes with a range whose output layer is tied to the embedding, and only with it")
    rows, data = [], list(outputs(files))
    held = {x["id"] for x in read(prompts) if set(x["ids"]) & changed} if tied else set()
    if tied and {i for (i, _, _), _, _ in data} - {x["id"] for x in read(prompts)}:
        raise SystemExit(f"an output whose prompt is not in {prompts}")
    for (i, s, _), (_, s2, _), (_, t, y) in data:
        k = first_difference(s, t)
        if events:
            inside = s != t and bool(y["events"]) and k >= min(y["events"])
        else:  # tied: a changed id both outputs share before k is also an input that changed, so the range is reached there
            inside = s != t and bool(set(s[k:k + 1] + t[k:k + 1] + (s[:k] if tied else [])) & changed)
        if inside:
            text = lambda xs: decode([x for x in xs if x is not STOP])  # noqa: E731
            print("FIRST DIFFERENCE IN THE RANGE", i, k, repr(text(s[k:k + 1])), "|", text(s[max(0, k - 10):k + 5]), "→", text(t[max(0, k - 10):k + 5]))
        rows.append({"id": i, "original twice the same": s == s2, "same": s == t, "first difference": None if s == t else k, "in the range": inside,
                     **({"changed id in the prompt": i in held} if tied else {})})
    checked = [r for r in rows if r["id"] not in held]
    totals = {"prompts": len(rows), "original twice the same": sum(r["original twice the same"] for r in rows), "same": sum(r["same"] for r in rows),
              "same or first difference in the range": sum(r["same"] or r["in the range"] for r in checked),
              **({"prompts with a changed id": len(rows) - len(checked)} if tied else {}),
              **({"events in the edited outputs": sum(len(y["events"]) for *_, (_, _, y) in data)} if events else
                 {"tokens of the range in the edited outputs": sum(t in changed for *_, (_, _, y) in data for t in y["tokens"])})}
    print("the same output", totals["same"], "| otherwise", [(r["id"], r["first difference"]) for r in checked if not (r["same"] or r["in the range"])],
          "|", {k: v for k, v in totals.items() if k.endswith("edited outputs") or k == "prompts with a changed id"})
    rng = {"events": "recorded in the edited outputs"} if events else {Path(range_file).name: sha(range_file)}
    finish(out, {"range": rng, **({"prompts": {Path(prompts).name: sha(prompts)}} if tied else {}), "generations": {Path(f).name: sha(f) for f in files},
                 "greedy": totals, "rows": rows})
