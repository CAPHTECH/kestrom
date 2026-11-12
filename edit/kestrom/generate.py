"""Greedy decoding (MLX) with the inference-time edits of a rules file (python -m pydoc kestrom.rules): logits edited where a rule
fires, directions added to layers where a condition on the prompt or the output holds, listed phrases backed out of, a loop in the
thinking stopped. Every output records its events, the positions where an edit acted; before the first one the decoding is the
original's, so check it with kestrom-verify compare events against two runs of the unchanged model (kestrom-verify generate mlx).
  kestrom generate MODEL PROMPTS.json OUT.jsonl [--rules RULES.dsl|FIT.npz] [--max-tokens N]
  --rules       the rules: a .dsl file, or a fit of kestrom rules fit when they read probed features
  --max-tokens  the longest output (512 unless given)
PROMPTS: a JSON list of {"ids": prompt ids, other keys kept}. OUT: per prompt, the keys of the prompt, finish (stop or length), the
tokens without the stop token, text, events (sorted positions) and detail (each backtrack: kind, position, kept: whether the final
output kept the tokens before it)."""
import argparse
import json

from kestrom_verify import report

from . import loopguard, rules

MAX_PER_POSITION = 20  # backtracks for a phrase at one position before giving up there


def start_token(tok, toks, char):
    """The index of the token whose text holds the character at char of decode(toks)."""
    j = len(toks) - 1
    while j > 0 and len(tok.decode(toks[:j])) > char:
        j -= 1
    return j


def processors(toks, ban, copy_found, think_end):
    """The logits processors of one restart at position len(toks): ban (phrases) at its first step only; after a loop was found,
    no 16-token piece the output already holds until </think>."""
    import mlx.core as mx
    out = []
    if ban:
        calls = []

        def first(tokens, logits):  # generate_step prefills the context without the processors; its first call is at len(toks)
            if not calls:
                logits[:, mx.array(ban)] = float("-inf")
            calls.append(1)
            return logits
        out.append(first)
    if copy_found:
        K, hist, nxt = loopguard.K, list(toks), {}
        state = {"calls": 0, "thinking": think_end not in toks}
        for i in range(K - 1, len(hist)):
            nxt.setdefault(tuple(hist[i - K + 1:i]), set()).add(hist[i])

        def forbid(tokens, logits):  # the last input token first, then one more token per call (the newest one)
            if state["calls"]:
                hist.append(int(tokens[-1]))
                nxt.setdefault(tuple(hist[-K:-1]), set()).add(hist[-1])
                state["thinking"] = state["thinking"] and hist[-1] != think_end
            state["calls"] += 1
            b = nxt.get(tuple(hist[-(K - 1):])) if state["thinking"] else None
            if b:
                logits[:, mx.array(sorted(b))] = float("-inf")
            return logits
        out.append(forbid)
    return out


def acted(record, p, end, prefill=False):
    """The output positions (from p, before end) whose logits an edit changed: the calls on one token are position p, then one more
    each (generate_step prefills the context but its last token first). With prefill (an add: what it adds to a layer reaches every
    later position), a prefill call that acted counts at p."""
    steps = [bool(f) for n, f in record if n == 1]
    return {p + j for j, f in enumerate(steps) if f and p + j < end} | ({p} if prefill and p < end and any(bool(f) for n, f in record if n > 1) else set())


def decode(model, tok, ids, max_tokens, phrases=(), window=None, think_end=None):
    """Greedy with the rules of rules.SIDE on the model and the watching edits given here (phrases, loopguard). Returns the tokens,
    the finish, the positions where an edit acted in the final output, and the backtracks."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    assert len(ids) > 2, "a prompt of more than two tokens (the record of the rules counts the calls after the prompt)"
    toks, bans, detail, fired = [], {}, [], set()
    copies, copy_found, thinking = (loopguard.Copies([]) if window else None), False, True
    while True:
        p = len(toks)
        rules.restart(len(ids))
        procs = processors(toks, sorted(bans.get(p, ())), copy_found, think_end)
        step = generate_step(mx.array(ids + toks), model, max_tokens=max_tokens - p, logits_processors=procs or None)
        done = len(tok.decode(toks)) if phrases else 0  # an occurrence counts once its end is past the text checked so far
        back, finish = None, "length"
        for t, _ in step:
            t = int(t)
            if t in tok.eos_token_ids:
                finish = "stop"
                break
            toks.append(t)
            if phrases:
                text = tok.decode(toks)
                hit = min(((i, s) for i, s in ((text.find(s, max(0, done - len(s) + 1)), s) for s in phrases) if i >= 0), default=None)
                done = len(text)
                if hit is not None:
                    q = start_token(tok, toks, hit[0])
                    if len(bans.get(q, ())) >= MAX_PER_POSITION:
                        detail.append(dict(kind="phrase", position=q, string=hit[1], banned=None, prefix=toks[:q], gave_up=True))
                    else:
                        detail.append(dict(kind="phrase", position=q, string=hit[1], banned=toks[q], prefix=toks[:q]))
                        bans = {k: v for k, v in bans.items() if k <= q}  # bans after q were made in another context
                        bans.setdefault(q, set()).add(toks[q])
                        back = q
            if back is None and copies is not None and not copy_found:
                copies.add(t)
                thinking = thinking and t != think_end
                q = copies.start(window) if thinking else None
                if q is not None:
                    detail.append(dict(kind="loop", position=q, prefix=toks[:q]))
                    copy_found, back = True, q
            if back is not None:
                break
        end = back if back is not None else len(toks) + (finish == "stop")
        fired = {j for j in fired if j < end} | acted(rules.SIDE["record"], p, end) | acted(rules.SIDE["record_add"], p, end, True)
        if back is None:
            return toks, finish, fired, detail
        del toks[back:]
        if copies is not None and not copy_found:
            copies, thinking = loopguard.Copies(toks), think_end not in toks


def run(model_dir, prompts_file, out, max_tokens=report.MAX_TOKENS, rules_file=None):
    from mlx_lm import load
    model, tok = load(str(model_dir))
    rules.SIDE.clear()
    if rules_file:
        rules.SIDE.update(rules.load(rules_file, tok))
        rules.attach(model)
    think_end = tok.convert_tokens_to_ids(loopguard.THINK_END) if loopguard.THINK_END in tok.get_vocab() else None
    with open(out, "w", encoding="utf-8") as f:  # one line per prompt as it is done (a long run keeps what it finished)
        for x in report.read(prompts_file):
            if rules_file:
                rules.start(model, x["ids"])
            toks, finish, fired, detail = decode(model, tok, x["ids"], int(max_tokens), rules.SIDE.get("phrases", ()), rules.SIDE.get("window"), think_end)
            for e in detail:
                e["kept"] = toks[:e["position"]] == e.pop("prefix")
            events = sorted(fired | {e["position"] for e in detail})
            row = dict({k: v for k, v in x.items() if k != "ids"}, finish=finish, tokens=toks, text=tok.decode(toks), events=events, detail=detail)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(x["id"], finish, len(toks), "events", len(events), flush=True)


def main(*argv):
    a = argparse.ArgumentParser(prog="kestrom generate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("model"), a.add_argument("prompts"), a.add_argument("out")
    a.add_argument("--max-tokens", type=int, default=report.MAX_TOKENS)
    a.add_argument("--rules")
    o = a.parse_args(argv)
    run(o.model, o.prompts, o.out, o.max_tokens, o.rules)
