"""Minimal end-to-end procedural memory KV demo with semantic routing and PREV_DIR update."""
import json
from pathlib import Path
import sys

from kestrom import kv, kv_multihop as kmh

EVAL_PROMPTS = [
    {
        "id": "cond",
        "cls": "cond_logic",
        "prompt": (
            "Given binary signals a=1, b=1, control c=1, and mask z=0, "
            "evaluate the conditional gate integer output y. "
            "Show brief steps on one line, then write 'Answer: <integer>' on the last line."
        ),
        "gold_init": "0",
        "gold_rev": "3",
    },
    {
        "id": "fee",
        "cls": "tiered_fee",
        "prompt": (
            "Calculate the final service charge in dollars when starting price is 120, "
            "loyalty member status is yes, and expedited handling is yes. "
            "Show brief steps on one line, then write 'Answer: <integer>' on the last line."
        ),
        "gold_init": "105",
        "gold_rev": "85",
    },
    {
        "id": "cipher",
        "cls": "cipher_rule",
        "prompt": (
            "Normalize the hyphenated code 'a1-mint-007' using the standard internal tag format. "
            "Show brief steps on one line, then write 'Answer: <result>' on the last line."
        ),
        "gold_init": "7_A1_tnim",
        "gold_rev": "7_A1_tnim",
    },
    {
        "id": "triage",
        "cls": "triage_policy",
        "prompt": (
            "Determine the escalation queue for an incident with severity=2, "
            "account_tier=vip, off_hours_weekend=yes, and retry_count=4. "
            "Show brief steps on one line, then write 'Answer: <RED|AMBER|BLUE|GREEN>' on the last line."
        ),
        "gold_init": "AMBER",
        "gold_rev": "AMBER",
    },
    {
        "id": "outside_1",
        "cls": "outside",
        "prompt": "What is the capital city of Portugal? Answer with the city name only.",
        "gold_init": None,
        "gold_rev": None,
    },
    {
        "id": "outside_2",
        "cls": "outside",
        "prompt": "Compute 17 + 28. Answer with the number only.",
        "gold_init": None,
        "gold_rev": None,
    },
]


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_plain(model, tok, prompt: str, max_tokens: int) -> dict:
    from mlx_lm.models.cache import make_prompt_cache

    ids = kv._chat_messages_ids(tok, [{"role": "user", "content": prompt}], enable_thinking=False)
    toks, finish, _ = kv._mlx_decode(model, tok, ids, max_tokens, make_prompt_cache(model))
    return {"tokens": toks, "finish": finish, "text": tok.decode(toks), "events": []}


def _run_routed(model, tok, bank, router, compose_cache, prompt: str, max_tokens: int) -> dict:
    entry_by_key = {e["key"]: e for e in bank["entries"]}
    ids = kv._chat_messages_ids(tok, [{"role": "user", "content": prompt}], enable_thinking=False)
    routed_key, _ = kmh.route_prompt(model, ids, router)
    if routed_key is None or routed_key not in entry_by_key:
        res = _run_plain(model, tok, prompt, max_tokens)
        res["routed"] = None
        return res
    sp = bank["strip_prefix"]
    cache = compose_cache.get_or_build(model, tok, bank, [entry_by_key[routed_key]])
    toks, finish, _ = kv._mlx_decode(model, tok, ids[len(sp):], max_tokens, cache)
    return {"tokens": toks, "finish": finish, "text": tok.decode(toks), "events": [0], "routed": routed_key}


def main(model_dir: str, out_dir: str):
    from mlx_lm import load

    out = Path(out_dir)
    model, tok = load(model_dir)
    bank_init = kv.load_bank(out / "bank_init", tok)
    bank_upd = kv.load_bank(out / "bank_upd", tok)
    router = kmh.load_router(out / "router.npz", model_dir=model_dir)

    base1, base2, routed_init = [], [], []
    c_init = kv.ComposeCache(maxsize=8)
    for item in EVAL_PROMPTS:
        mt = 24 if item["cls"] == "outside" else 160
        r1 = _run_plain(model, tok, item["prompt"], mt)
        r2 = _run_plain(model, tok, item["prompt"], mt)
        rk = _run_routed(model, tok, bank_init, router, c_init, item["prompt"], mt)
        base1.append({"id": item["id"], "finish": r1["finish"], "tokens": r1["tokens"], "text": r1["text"]})
        base2.append({"id": item["id"], "finish": r2["finish"], "tokens": r2["tokens"], "text": r2["text"]})
        routed_init.append({"id": item["id"], "finish": rk["finish"], "tokens": rk["tokens"], "text": rk["text"], "events": rk["events"]})
        print(f"[init] {item['id']} routed={rk['routed']} expected={item['gold_init']} -> {rk['text'].splitlines()[-1] if rk['text'].strip() else ''}")

    _write_jsonl(out / "base1.jsonl", base1)
    _write_jsonl(out / "base2.jsonl", base2)
    _write_jsonl(out / "kv_routed.jsonl", routed_init)

    c_upd = kv.ComposeCache(maxsize=8)
    for item in EVAL_PROMPTS[:4]:
        ru = _run_routed(model, tok, bank_upd, router, c_upd, item["prompt"], 160)
        print(f"[upd]  {item['id']} routed={ru['routed']} expected={item['gold_rev']} -> {ru['text'].splitlines()[-1] if ru['text'].strip() else ''}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
