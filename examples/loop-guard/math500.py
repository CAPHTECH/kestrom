"""MATH-500 for the loop-guard example: the prompts (thinking on, the instruction of the Astra model card) and the score (the last
\\boxed{} checked with math-verify).
Usage: python math500.py prompts MODEL MATH500.jsonl OUT.json [N]   (the odd-index problems, the first N; 100 unless given)
       python math500.py score OUT.jsonl [...]"""
import json
import sys

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def prompts(model, math_file, out, n=100):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(line) for line in open(math_file, encoding="utf-8")][1::2][:int(n)]
    data = [dict(id=r["unique_id"], answer=r["answer"], ids=tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": r["problem"] + SUFFIX}], add_generation_prompt=True, tokenize=False, enable_thinking=True),
        add_special_tokens=False)) for r in rows]
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(len(data), "prompts")


def boxed(text):
    """The content of the last \\boxed{...} (braces balanced), else None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def score(*files):
    from math_verify import parse, verify
    for f in files:
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        ok = [(b := boxed(r["text"])) is not None and verify(parse("$" + r["answer"] + "$"), parse("$" + b + "$")) for r in rows]
        cut = sum(r["finish"] == "length" for r in rows)
        print(f, json.dumps({"outputs": len(rows), "correct": sum(ok), "cut at the limit": cut, "outputs with an event": sum(bool(r.get("events")) for r in rows)}))


if __name__ == "__main__":
    {"prompts": prompts, "score": score}[sys.argv[1]](*sys.argv[2:])
