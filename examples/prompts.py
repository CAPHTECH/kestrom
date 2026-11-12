"""Prompts for kestrom-verify and kestrom: one question per line of a text file, through the model's chat template.
Usage: python examples/prompts.py MODEL QUESTIONS.txt OUT.json [think]   (think: the chat template's thinking on; off unless given)"""
import json
import sys


def main(model, questions, out, think=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    lines = [x for x in open(questions, encoding="utf-8").read().split("\n") if x.strip()]
    rows = [dict(id=i, ids=tok.encode(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                                                enable_thinking=think == "think"), add_special_tokens=False))
            for i, q in enumerate(lines)]
    json.dump(rows, open(out, "w", encoding="utf-8"))
    print(len(rows), "prompts")


if __name__ == "__main__":
    main(*sys.argv[1:])
