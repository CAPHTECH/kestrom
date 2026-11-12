"""End to end with a real model (MLX), skipped unless KESTROM_MODEL is a Hugging Face folder of a chat model whose output layer is
not tied to the embedding (we run it with Qwen/Qwen3.5-9B): a row ban checked by range (twice the same bytes), logits and compare;
the loop guard checked by compare events."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

MODEL = os.environ.get("KESTROM_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="KESTROM_MODEL is not set")
QUESTIONS = ["What is the capital of France?", "List three prime numbers.", "日本で一番高い山は？"]


def run(*args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(["verify", "edit"]))
    out = subprocess.run([sys.executable, "-B", "-m", *args], capture_output=True, text=True, env=env,
                         cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def checks(stdout):
    return [line for line in stdout.splitlines() if line.endswith(("PASS", "FAIL"))]


@pytest.fixture(scope="module")
def prompts(tmp_path_factory):
    from transformers import AutoTokenizer
    t = tmp_path_factory.mktemp("model")
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [dict(id=i, ids=tok.encode(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                                                enable_thinking=False), add_special_tokens=False)) for i, q in enumerate(QUESTIONS)]
    (t / "prompts.json").write_text(json.dumps(rows))
    for name in ("base", "base2"):
        run("kestrom_verify", "generate", "mlx", MODEL, str(t / "prompts.json"), str(t / f"{name}.jsonl"), "48")
    return t


def test_ban(prompts):
    t = prompts
    first = json.loads((t / "base.jsonl").read_text().splitlines()[0])["tokens"]
    (t / "list.txt").write_text(f"id:{first[3]}\nid:{first[5]}\n")  # two tokens the original writes in its first answer
    from kestrom_verify import weights
    file, start, end, n, _ = weights.output(MODEL)
    width = (end - start) // n // 2
    np.save(t / "row.npy", np.full(width, -0.01, np.float32))
    run("kestrom", "rows", "ban", MODEL, MODEL, str(t / "list.txt"), str(t / "row.npy"), str(t / "edited"))
    for i in (1, 2):
        assert all(c.endswith("PASS") for c in checks(run("kestrom_verify", "range", MODEL, str(t / "edited"), str(t / f"range{i}.json"))))
    assert (t / "range1.json").read_bytes() == (t / "range2.json").read_bytes()
    assert json.loads((t / "range1.json").read_text())["weights"]["changed rows"] == sorted({first[3], first[5]})
    run("kestrom_verify", "generate", "mlx", str(t / "edited"), str(t / "prompts.json"), str(t / "edited.jsonl"), "48")
    out = run("kestrom_verify", "compare", str(t / "range1.json"), str(t / "base.jsonl"), str(t / "base2.jsonl"), str(t / "edited.jsonl"), MODEL, str(t / "greedy.json"))
    assert all(c.endswith("PASS") for c in checks(out)) and len(checks(out)) == 2
    assert json.loads((t / "greedy.json").read_text())["greedy"]["same"] < len(QUESTIONS)  # the first answer must change
    inputs = [dict(id=json.loads(line)["id"], ids=p["ids"], tokens=json.loads(line)["tokens"])
              for line, p in zip((t / "base.jsonl").read_text().splitlines(), json.loads((t / "prompts.json").read_text()))]
    (t / "inputs.json").write_text(json.dumps(inputs))
    out = run("kestrom_verify", "logits", MODEL, str(t / "edited"), str(t / "inputs.json"), str(t / "logits.json"))
    assert all(c.endswith("PASS") for c in checks(out)) and len(checks(out)) == 3


def test_loopguard_keeps_the_original_until_the_event(prompts):
    t = prompts
    (t / "guard.dsl").write_text("loopguard: 32\n")
    run("kestrom", "generate", MODEL, str(t / "prompts.json"), str(t / "guard.jsonl"), "--max-tokens", "48", "--rules", str(t / "guard.dsl"))
    out = run("kestrom_verify", "compare", "events", str(t / "base.jsonl"), str(t / "base2.jsonl"), str(t / "guard.jsonl"), MODEL, str(t / "g.json"))
    assert all(c.endswith("PASS") for c in checks(out)) and len(checks(out)) == 2
