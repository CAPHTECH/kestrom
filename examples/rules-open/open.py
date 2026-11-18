"""The router of examples/rules-run opening a model's answers (examples/rules-open/README.md).
  python open.py prompts MODEL OUT.json      the eight questions of ../rules-run/router.json through the chat template, each with "data"
  python open.py chat MODEL OUT.json         four questions that ask for no class, with no system prompt
  python open.py score FILE.jsonl ...        per file: how many answers open with the right line (the label's meter, ask with none)"""
import json
import sys

SYSTEM = """You sort questions about a building's electricity meters. Answer in this exact form.
Line 1: exactly one of these words and nothing else: ac (air conditioning), light (lighting), ask (neither, both, or unclear).
Line 2: one short sentence in Japanese saying why."""
QUESTIONS = json.load(open("../rules-run/router.json", encoding="utf-8"))
CHAT = ["今日の天気はどうかな", "Pythonでリストを逆順にする方法を一行で教えて", "照明デザイナーという仕事について一文で教えて", "エアコンという言葉の語源を一文で教えて"]


def prompts(model, out, system=SYSTEM, questions=QUESTIONS):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    rows = [dict(id=x["id"], data={"text": x["text"]}, ids=tok.encode(tok.apply_chat_template(
        [{"role": "system", "content": system}] * bool(system) + [{"role": "user", "content": x["text"]}], add_generation_prompt=True, tokenize=False,
        enable_thinking=False), add_special_tokens=False)) for x in questions]
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(len(rows), "prompts")


def score(*files):
    gold = {x["id"]: (x["label"] or {}).get("meter") or "ask" for x in QUESTIONS}
    for f in files:
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        first = {r["id"]: r["text"].split("\n")[0] for r in rows}
        print(f, "first line right", sum(first[i] == g for i, g in gold.items()), "of", len(gold),
              "| wrong", {i: first[i] for i, g in gold.items() if first[i] != g}, "| with events", sum(bool(r.get("events")) for r in rows))


def chat(model, out):
    prompts(model, out, None, [{"id": i, "text": q} for i, q in enumerate(CHAT)])


if __name__ == "__main__":
    {"prompts": prompts, "chat": chat, "score": score}[sys.argv[1]](*sys.argv[2:])
