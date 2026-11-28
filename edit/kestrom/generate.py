"""Greedy decoding (MLX) with the inference-time edits of a rules file (python -m pydoc kestrom.rules): logits edited where a rule
fires, directions added to layers where a condition on the prompt or the output holds, listed phrases backed out of, a loop in the
thinking stopped, the answer opened with the class that rules of their own answer for the prompt (and the thinking with a text that holds it). Every output records its events, the positions where an edit acted; before the first one the decoding is the
original's, so check it with kestrom-verify compare events against two runs of the unchanged model (kestrom-verify generate mlx).
  kestrom generate MODEL PROMPTS.json OUT.jsonl [--rules RULES.dsl|FIT.npz] [--max-tokens N]
  --rules       the rules: a .dsl file, or a fit of kestrom rules fit when they read probed features
  --max-tokens  the longest output (512 unless given)
PROMPTS: a JSON list of {"ids": prompt ids, "data" (with an open or kv line: what its rules read), other keys kept}. OUT: per prompt, the keys of the prompt, finish (stop or length),
the tokens without the stop token, text, events (sorted positions) and detail (each backtrack: kind, phrase or loop, position,
kept: whether the final output kept the tokens before it; a phrase also string and banned, the token banned, null with gave_up
when it gave up), with an open line open (the class the answer opens with by that line, null where the rules answered none
and the line gives no class, the answer was left alone, the thinking did not end or the answer was cut short of it), with a kv line
kv (the matched bank entry key, a list of keys when MAX_MATCH > 1 matches several, or null where no entry matched), and with an alu line
alu (each inline sidecar arithmetic firing whose whole value the output holds: position, expr, injected)."""
import argparse
import json
import re

import numpy as np

from kestrom_verify import report

from . import kv, loopguard, rules

MAX_PER_POSITION = 20  # backtracks for a phrase at one position before giving up there
INLINE_ALU_RE = re.compile(r"(?<![0-9A-Za-z_.,])(\d{1,3}(?:,\d{3})+|\d+)\s*([+\-*×])\s*(\d{1,3}(?:,\d{3})+|\d+)\s*=( ?)\Z")
ALU_BEFORE = tuple("+-−*×/÷^%(")  # an lhs after one of these is part of a longer expression (or signed): not computed


def match_inline_alu(text, min_digits):
    """If `text` ends with a qualifying multi-digit `<lhs> <op> <rhs> =` expression, return `(expr, inject_str)` else `(None, None)`:
    not where the lhs continues a number (a digit, a letter, `.` or `,` before it) or an expression (an operator or `(` before it)."""
    m = INLINE_ALU_RE.search(text)
    if not m or text[:m.start()].rstrip()[-1:] in ALU_BEFORE:
        return None, None
    lhs_s, op, rhs_s, trailing_space = m.group(1).replace(",", ""), m.group(2), m.group(3).replace(",", ""), m.group(4)
    is_mul = op in ("*", "×")
    if not ((len(lhs_s) >= min_digits or len(rhs_s) >= min_digits) or (is_mul and len(lhs_s) >= 3 and len(rhs_s) >= 3)):
        return None, None
    lhs, rhs = int(lhs_s), int(rhs_s)
    val = lhs + rhs if op == "+" else lhs - rhs if op == "-" else lhs * rhs
    return f"{lhs}{op}{rhs}", ("" if trailing_space == " " else " ") + str(val)


def start_token(tok, toks, char):
    """The index of the token whose text holds the character at char of decode(toks)."""
    j = len(toks) - 1
    while j > 0 and len(tok.decode(toks[:j])) > char:
        j -= 1
    return j


def spell(tok, out, opening, top, order):
    """The token after the tokens `out` of the current part (the thinking, or the answer) while the text is short of `opening`: the top token where the text with it still
    fits (in whole characters, as far as opening goes, and longer than before), else the first of order() that does, and whether it
    is another than top. It stops where none does (also where the other edits ban every token that would)."""
    n = len(tok.decode(out))

    def fits(t):
        s = tok.decode(out + [t])
        return len(s) > n and "\ufffd" not in s[:len(opening)] and (opening.startswith(s) or s.startswith(opening))
    if fits(top):
        return top, False
    t = next((int(t) for t in order() if fits(int(t))), None)
    if t is None:
        raise SystemExit(f"no token goes on from {tok.decode(out)!r} toward {opening!r} in whole characters")
    return t, True


def answer(tok, toks, think, thinking=False, part=()):
    """Where the tokens toks leave a decoding that was (thinking, with the tokens `part` of its current part): thinking from <think>
    to </think> (think: their ids), and the part the tokens after the last of the two (the thinking, or the answer) but for
    white-space tokens before its first other one."""
    part = list(part)
    for t in toks:
        if t in think:
            thinking, part = t == think[0], []
        elif part or tok.decode([t]).strip():
            part.append(t)
    return thinking, part


def processors(toks, ban, copy_found, think_end, tok=None, opening=None, spelled=None, think=(None, None), thinking=False, alu=None, alu_fires=None):
    """The logits processors of one restart at position len(toks): ban (phrases) at its first step only; after a loop was found,
    no 16-token piece the output already holds until </think>; with an opening (an open line: its characters, those of every
    class the answer may open with, and those the thinking opens with or None), its characters while the answer (see answer;
    thinking: whether the prompt left thinking open) is short of them and has gone their way, where the top token that begins the
    answer starts one of the classes, and the thinking's characters while it is short of them; the positions where another token
    than the top one was taken are added to spelled."""
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
    if opening:
        opening, starts, note = opening
        seen, opened = list(toks), []
        opened_ = {"done": False, "at": answer(tok, toks, think, thinking)}  # its own name: the processors above keep theirs

        def open_(tokens, logits):  # after the others: the top token is the one the other edits leave, and a token they ban is not taken
            if opened:
                seen.append(int(tokens[-1]))
                opened_["at"] = answer(tok, seen[-1:], think, *opened_["at"])
                opened_["done"] = opened_["done"] and seen[-1] not in think  # a new part: the thinking, or the answer
            opened.append(1)
            now_thinking, part = opened_["at"]
            target = note if now_thinking else opening
            if opened_["done"] or target is None:
                return logits
            top, text = int(mx.argmax(logits[0])), tok.decode(part)
            if not text:
                s = tok.decode([top]).lstrip()
                if top in think or not s:  # the part has not begun or ends empty: <think>, </think>, white space
                    return logits
                opened_["done"] = not now_thinking and not any(c.startswith(s) or s.startswith(c) for c in starts)  # not a class: left alone
            opened_["done"] = opened_["done"] or len(text) >= len(target) or not target.startswith(text)
            if opened_["done"]:
                return logits
            z = np.array(logits[0].astype(mx.float32))
            t, other = spell(tok, part, target, top, lambda: (t for t in np.argsort(-z, kind="stable") if z[t] > -np.inf))
            if other:
                spelled.append(len(seen))
                logits = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
                logits[:, t] = 0.0
            return logits
        out.append(open_)
    if alu is not None:
        alu_seen, alu_pending, alu_called = list(toks), [], []

        def alu_(tokens, logits):  # after the others: a token they ban is not taken, and the firing is given up
            if alu_called:
                alu_seen.append(int(tokens[-1]))
            alu_called.append(1)
            if not alu_pending and alu_seen:
                text = tok.decode(alu_seen)
                expr, inject = match_inline_alu(text, alu)
                if inject is not None:
                    ids_inj = [int(t) for t in tok.encode(inject, add_special_tokens=False)]
                    if ids_inj and tok.decode(alu_seen + ids_inj) == text + inject:  # the tokens spell the value in this context
                        alu_pending.extend(ids_inj)
                        alu_fires.append({"position": len(alu_seen), "expr": expr, "injected": inject.strip(), "length": len(ids_inj)})
            if alu_pending and float(logits[0, alu_pending[0]]) == float("-inf"):
                alu_pending.clear()
                alu_fires.pop()
            if alu_pending:
                t = alu_pending.pop(0)
                spelled.append(len(alu_seen))
                logits = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
                logits[:, t] = 0.0
            return logits
        out.append(alu_)
    return out


def acted(record, p, end, prefill=False):
    """The output positions (from p, before end) whose logits an edit changed: the calls on one token are position p, then one more
    each (generate_step prefills the context but its last token first). With prefill (an add: what it adds to a layer reaches every
    later position), a prefill call that acted counts at p."""
    steps = [bool(f) for n, f in record if n == 1]
    return {p + j for j, f in enumerate(steps) if f and p + j < end} | ({p} if prefill and p < end and any(bool(f) for n, f in record if n > 1) else set())


def decode(model, tok, ids, max_tokens, phrases=(), window=None, think_end=None, opening=None, think_start=None, cache_fn=None, alu=None, alu_fires=None):
    """Greedy with the rules of rules.SIDE on the model and the watching edits given here (phrases, loopguard, opening, cache_fn, alu). Returns the tokens,
    the finish, the positions where an edit acted in the final output, and the backtracks."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    assert len(ids) > 2, "a prompt of more than two tokens (the record of the rules counts the calls after the prompt)"
    toks, bans, detail, fired = [], {}, [], set()
    think = (think_start, think_end)
    thinking0 = bool(opening) and answer(tok, ids, think)[0]
    copies, copy_found, thinking = (loopguard.Copies([]) if window else None), False, True
    while True:
        p = len(toks)
        rules.restart(len(ids))
        spelled = []
        round_alu = []
        procs = processors(toks, sorted(bans.get(p, ())), copy_found, think_end, tok, opening, spelled, think, thinking0, alu, round_alu)
        prompt_cache = cache_fn() if cache_fn is not None else None
        step = generate_step(mx.array(ids + toks), model, max_tokens=max_tokens - p, logits_processors=procs or None, prompt_cache=prompt_cache)
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
        fired = {j for j in fired if j < end} | acted(rules.SIDE["record"], p, end) | acted(rules.SIDE["record_add"], p, end, True) | {j for j in spelled if j < end}
        if alu is not None:  # a firing counts where its whole value is in the output
            alu_fires[:] = [e for e in alu_fires + round_alu if e["position"] + e["length"] <= end]
        if back is None:
            if cache_fn is not None:
                fired.add(0)
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
        if rules.SIDE["kv"]:
            kv.check_model(rules.SIDE["kv"][0], model_dir)
        rules.attach(model)
    compose_cache = kv.ComposeCache() if rules.SIDE.get("kv") else None
    think_end = tok.convert_tokens_to_ids(loopguard.THINK_END) if loopguard.THINK_END in tok.get_vocab() else None
    think_start = tok.convert_tokens_to_ids("<think>") if "<think>" in tok.get_vocab() else None
    with open(out, "w", encoding="utf-8") as f:  # one line per prompt as it is done (a long run keeps what it finished)
        for x in report.read(prompts_file):
            if rules_file:
                rules.start(model, x["ids"])
            opening = rules.opening(x)
            kv_hit = rules.kv_for(x, x["ids"]) if rules_file else None
            if kv_hit is not None:
                kv_key, kv_entries, strip_prefix = kv_hit
                bank = rules.SIDE["kv"][0]
                feed_ids = x["ids"][len(strip_prefix):]
                cache_fn = lambda entries=kv_entries, b=bank: compose_cache.get_or_build(model, tok, b, entries)  # noqa: E731
            else:
                kv_key, feed_ids, cache_fn = None, x["ids"], None
            alu_fires = [] if rules.SIDE.get("alu") is not None else None
            toks, finish, fired, detail = decode(model, tok, feed_ids, int(max_tokens), rules.SIDE.get("phrases", ()), rules.SIDE.get("window"), think_end, opening, think_start, cache_fn, rules.SIDE.get("alu"), alu_fires)
            for e in detail:
                e["kept"] = toks[:e["position"]] == e.pop("prefix")
            events = sorted(fired | {e["position"] for e in detail})
            row = dict({k: v for k, v in x.items() if k != "ids"}, finish=finish, tokens=toks, text=tok.decode(toks), events=events, detail=detail)
            if rules.SIDE.get("open"):
                thinking, part = answer(tok, toks, (think_start, think_end), answer(tok, x["ids"], (think_start, think_end))[0])
                row["open"] = opening[0][:-1] if opening and not thinking and tok.decode(part).startswith(opening[0]) else None
            if rules.SIDE.get("kv"):
                row["kv"] = kv_key
            if rules.SIDE.get("alu") is not None:
                row["alu"] = [{k: v for k, v in e.items() if k != "length"} for e in alu_fires]
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
