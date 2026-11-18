"""Rules written by hand in a small language, run at decoding time (MLX, greedy), or with no model (run, build, weigh). The inputs of the
rules are features of the hidden state, each read by a linear probe, or conditions on the prompt and on the output so far; units
combine them; the action lines raise, move or ban token logits, add a direction to a layer's output where a unit fires, go back at a
listed phrase, or stop a loop in the thinking. Where no unit of an action fires, that action leaves the computation as it was
(logits taken back bit for bit, nothing added), so the output is the original's until the first event (kestrom-verify compare
events; the events are the positions where an action acted).
  kestrom generate MODEL PROMPTS.json OUT.jsonl --rules RULES.dsl|FIT.npz [--max-tokens N]
      greedy decoding with the rules (python -m pydoc kestrom.generate); rules that read probed features need a fit
  kestrom rules fit MODEL RULES.dsl DATA.json OUT.npz [BALANCE] [K] [STEPS] [LR]
      DATA: a JSON list of {"ids": prompt ids, "tokens": answer ids, "features": {name: [j, ...]}}: the answer positions j where each
      probed input holds (the position that predicts tokens[j]). Every probed input gets a ridge probe on the input of the output layer
      at every answer position (each probe's positions weigh BALANCE times the rest in total, 1 unless given). K (4 unless given)
      scales what the out lines add to the logits and the margin of the moves. With STEPS (0 unless given; not with `in` lines), the
      probes are then tuned by gradient (Adam, full batch, the step LR times the spread of the probes, 0.001 unless given) on the
      cross entropy of the answer tokens through the out and move lines (the units continuous, no gate); the rules stay as written.
  kestrom rules write-mlp MODEL RULES.dsl DATA.json DST [BALANCE] [K]
      rules of h1, h2 and out lines only (one layer of units) written into the last MLP instead (tier 2; python -m pydoc kestrom.mlp).
  kestrom rules run RULES.dsl DATA.json OUT.jsonl
      the rules as a network of their own, with no model (no tier: nothing is edited), for a trial of rules or a small classifier
      or state machine written by hand. Rules of in lines of match, count, number and prev, h1, h2, out and set lines only, with as many
      units as they name (with a model, 32 a layer); the out lines name classes, not tokens. DATA: a JSON list of {"id", "text" (read by the
      match, count and number lines), "values": {name: number} (every input that no in line declares), "label" (optional; null: no
      answer is right; with heads, {head: class}), "targets": {register: number} (optional)}, or with "steps": [{"text", "values"}, ...] in place of text and values (or "steps": N beside them: the
      entry itself N times): the network runs once per step, the prev lines reading which units fired at the step before (none
      before the first), and the entry's answer is the last step's. A set line keeps a number in a register along the steps: it
      starts at the value of its name in the first step (0 without) and is read as an input by that name; after each step the set
      lines whose unit fired write their registers, all from the numbers read at that step (the last line that fires wins).
      A register after one step is a numeric output (`set y: 2*temp + 1 if warm`); with targets it prints each register's mean
      absolute error. Classes named HEAD/CLASS are answered per head: "class" is then {head: class}, each head null on its own; an entry is right
      when every head of its label is, wrong when a head answers another class, and has no answer otherwise. A unit that fires counts as 1, so a class scores the sum of its coefficients of the units that fire (rounded to 6
      decimals); OUT.jsonl has per entry {"id", "class", "scores", "fired" (the h2 units of the out lines that fire), "steps" (the
      class of every step, for an entry with steps), "registers" (their numbers after the last step)}, class the top score, also where none of its own units fires (a negative
      coefficient elsewhere), and null where no unit fires or the top score is tied. With labels it prints how many are right, have
      no answer, and the ids answered wrong. The units are written for inputs of 0 and 1, so a gate holds from its threshold minus
      0.5 (`temp >= 30` from 29.5): give values in whole numbers (below 2^24: the network computes in float32), and read a value, a count or a number through `K*x >= T`, not all or any. The regexes search the text as given (Python re, no normalization)
  kestrom rules build RULES.dsl NET.npz
      the same rules written out as the network: inputs, kind and arg (of each input: value, or match, count, number with its
      regex, or prev with its h2 unit), units (the names of the h2 units), classes, W1, b1, W2, b2, W3, b3, and for the set lines set_register,
      set_expression and set_unit (the h2 unit, by index). h1 = tanh(x W1 + b1), h2 = tanh(h1 W2 + b2), a unit fires where h2 >= 0,
      scores = (fires ? 1 : −1) W3 + b3
  kestrom rules weigh RULES.dsl DATA.json OUT.dsl [RIDGE]
      the rules with their out lines rewritten: the coefficients of every class over all the h2 units, by least squares (ridge
      RIDGE, 0.001 unless given) toward 1 on the entries labelled with the class and 0 on the others, from which units fire at each
      labelled entry's last step. The other lines are kept as written, the new out lines after them; the coefficients are rounded to 4 decimals.
The language: one line each, `HEAD: BODY` (the first : outside double quotes); `#` outside double quotes starts a comment; TEXT is
a JSON string ("..."); a TOKEN is a word or a JSON string ("##", " the"); paths as given (from the working directory). An input is
a name in an h1 line: declared by an `in` line, or else read by a probe (by run, build and weigh: a value given in DATA, or a register).
  in NAME: prompt FILE LAYER                 holds on the whole decoding when the mean over the prompt tokens of LAYER's output,
                                             on the condition vector of FILE (kestrom steer direction), is past its threshold
  in NAME: after TEXT                        holds from the token that completes the first TEXT in the output on
  in NAME: thinking                          holds where the last of <think> and </think> so far is <think>
  in NAME: position I [J]                    holds where the next token is at output position I to J (from 0, the first token
                                             after the prompt, as events count; J = I unless given): fields at fixed positions
  in NAME: match TEXT                        run, build and weigh only, as the next three: holds where the regex TEXT (a JSON string:
                                             "\\\\d+" for \\d+) is found in the text
  in NAME: count TEXT                        the number of matches of the regex TEXT in the text (not overlapping)
  in NAME: number TEXT                       the number in the one group of the regex TEXT at its first match ("(\\\\d+)階"), 0 with none
  in NAME: prev UNIT                         holds where the h2 unit UNIT fired at the step before: state carried along the steps
  set NAME: (a - 2*b) % c + 1 if UNIT       the register NAME takes this number, computed with + - * / // % and parentheses over
                                             inputs, registers and numbers (names without a dot that are not Python keywords), after a step where the h2 unit
                                             UNIT fires (run, build and weigh only)
  h1 NAME: all(a, !b, ...) | any(a, ...) | 2*a - b >= 1      a unit over the inputs (fires when the gate holds)
  h2 NAME: (the same, over h1 units)                         a unit over the h1 units; the actions name h2 units
  out TOKEN: 4*u + 2*v                       raise the logit of TOKEN (one token, one out line each) by K times the h2 units that fire (each 0 to 1)
  move X -> Y: u                             where u fires (u from 0 to 1), Y takes the logit of X plus K and X that of Y minus K:
                                             d = max(0, z_X − z_Y + K), z_Y += u·d, z_X −= u·d (so Y comes first only where X came
                                             first, or within K; where Y is already K above X, nothing moves); a token in one move only
  ban TOKEN ...: u                           the logits of the tokens (one token each) set to −inf where u fires
  add FILE LAYER: 8*u + 2*v                  the coefficients of the h2 units that fire, times the unit direction of LAYER in FILE (its
                                             "direction": kestrom find direction or kestrom steer direction), added to LAYER's
                                             output at those positions (also over the prompt). Its units read only `in` inputs: a
                                             probe reads the last layer, which a layer's output comes before
  phrase: TEXT                               once TEXT is complete in the output, the decoding goes back to the token where it starts,
                                             bans that token there and goes on (the Antislop sampler, arXiv 2510.15061; at most 20
                                             times at one position)
  loopguard: [WINDOW]                        at the first copy in the thinking (the last WINDOW tokens, 512 unless given, made of
                                             16-token pieces that all occurred earlier), the decoding goes back to its start and from
                                             there to </think> completes no 16-token piece the output already holds (kestrom.loopguard)
"""
import ast
import json
import operator
import re

import numpy as np

from kestrom_verify import report

from .loopguard import WINDOW
from .rows import trunk

GAIN, UNITS = 6.0, 32  # a unit is tanh(GAIN·(gate − threshold + 0.5)): about +1 when its gate holds, −1 when it does not
LAMBDA, GATE = 1e-2, 0.5  # the ridge of the probes (times the mean diagonal of the Gram matrix); a unit fires at (h2 + 1)/2 ≥ GATE
K = 4.0  # the scale of the out lines and the margin of the moves, unless a fit gives another
TERM = re.compile(r"([+-]?)\s*(?:(\d+(?:\.\d+)?)\s*\*\s*)?([A-Za-z_][\w.]*)")
CODE = re.compile(r'(?:[^"#]|"(?:[^"\\]|\\.)*")*')  # a line up to a # outside double quotes
HEAD = re.compile(r'((?:[^":]|"(?:[^"\\]|\\.)*")*):')  # the head: up to the first : outside double quotes
KINDS = ("in", "h1", "h2", "out", "move", "ban", "add", "phrase", "loopguard", "set")


def lines(text):
    """[(kind, the rest of the head, body)] of the lines that are not empty or comments."""
    out = []
    for r in text.splitlines():
        r = CODE.match(r).group(0).strip()
        if not r:
            continue
        m = HEAD.match(r)
        head, body = (m.group(1), r[m.end():]) if m else ("", "")
        words = head.split(None, 1)
        if not m or not words or words[0] not in KINDS:
            raise ValueError(f"every line is `KIND ...: ...` with KIND one of {', '.join(KINDS)}: {r!r}")
        out.append((words[0], words[1].strip() if len(words) > 1 else "", body.strip()))
    return out


def terms(expr):
    """[(name, coefficient)] of 'a + 2*b - c'; anything else stops (also two terms with no sign between them)."""
    t = r"(?:\d+(?:\.\d+)?\s*\*\s*)?[A-Za-z_][\w.]*"
    if not re.fullmatch(rf"\s*[+-]?\s*{t}(?:\s*[+-]\s*{t})*\s*", expr):
        raise ValueError(f"cannot read {expr!r} as a sum of terms like a + 2*b - c")
    return [(name, (-1 if sign == "-" else 1) * float(coef or 1)) for sign, coef, name in TERM.findall(expr)]


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
       ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def names_of(expr):
    """The names a set expression reads; it stops unless the expression is + - * / // % and parentheses over names and numbers."""
    try:
        nodes = list(ast.walk(ast.parse(expr.strip(), mode="eval")))
    except SyntaxError:
        nodes = [None]
    if not all(isinstance(n, (ast.Expression, ast.Name, ast.Load, *OPS)) or (isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPS)
               or (isinstance(n, ast.Constant) and type(n.value) in (int, float)) for n in nodes):
        raise ValueError(f"a set line computes with + - * / // % and parentheses over names and numbers: {expr!r}")
    return [n.id for n in nodes if isinstance(n, ast.Name)]


def value(expr, read):
    """The number of a set expression (checked by names_of), read giving the number of each name."""
    def v(n):
        if isinstance(n, ast.BinOp):
            return OPS[type(n.op)](v(n.left), v(n.right))
        return OPS[type(n.op)](v(n.operand)) if isinstance(n, ast.UnaryOp) else read[n.id] if isinstance(n, ast.Name) else float(n.value)
    return v(ast.parse(expr.strip(), mode="eval").body)


def gate(body):
    """'all(a,!b)', 'any(a,b)' or 'linear >= t' -> ([(name, weight)], threshold)."""
    body = body.strip()
    m = re.fullmatch(r"(all|any)\((.*)\)", body)
    if m:
        pairs = [(i.strip().lstrip("!"), -1.0 if i.strip().startswith("!") else 1.0) for i in m.group(2).split(",")]
        return pairs, float(sum(w > 0 for _, w in pairs)) if m.group(1) == "all" else float(1 - sum(w < 0 for _, w in pairs))
    left, threshold = body.rsplit(">=", 1)
    return terms(left), float(threshold)


def words(rest):
    """The tokens of an out or ban head: words, or JSON strings ("##", "\\n", " the")."""
    return [json.loads(w) if w.startswith('"') else w for w in re.findall(r'"(?:[^"\\]|\\.)*"|\S+', rest)]


def text_of(s):
    if not (s.startswith('"') and s.endswith('"')):
        raise ValueError(f"not a JSON string: {s}")
    return json.loads(s)


def declare(body):
    """The spec of an `in` line: ("prompt", FILE, LAYER), ("after", TEXT), ("thinking",), ("position", I, J), ("match" | "count" | "number", REGEX) or
    ("prev", h2 unit)."""
    w = body.split(None, 1)
    if w and w[0] == "prompt" and len(body.split()) == 3:
        return ("prompt", body.split()[1], int(body.split()[2]))
    if w and w[0] == "after" and len(w) == 2:
        return ("after", text_of(w[1].strip()))
    if body == "thinking":
        return ("thinking",)
    p = body.split()
    if p[:1] == ["position"] and len(p) in (2, 3) and all(x.isdigit() for x in p[1:]) and int(p[1]) <= int(p[-1]):
        return ("position", int(p[1]), int(p[-1]))
    if w and w[0] in ("match", "count", "number") and len(w) == 2:
        r = re.compile(text_of(w[1].strip()))
        if w[0] == "number" and r.groups != 1:
            raise ValueError(f"the regex of a number line has one group, the number: {body!r}")
        return (w[0], r.pattern)
    if p[:1] == ["prev"] and len(p) == 2:
        return ("prev", p[1])
    raise ValueError("an in line is `prompt FILE LAYER`, `after TEXT`, `thinking`, `position I [J]` (0 <= I <= J), `match REGEX`, `count REGEX`, "
                     f"`number REGEX` or `prev UNIT`: {body!r}")


def compile_rules(text, units=UNITS):
    """The rules as a dict: inputs (the probed ones, sorted, then those of the in lines in order), declared {name: spec}, tokens (of
    the out lines), moves [(X, Y, h2 unit)], bans [(tokens, h2 unit)], adds [(FILE, LAYER, [(h2 unit, coefficient)])], phrases, window
    (of loopguard, None without), used (the h2 units of the out, move and ban lines), units (the names of the h2 units), sets [(register, expression, h2 unit)] and net {W1,
    b1, W2, b2, W3, b3}: out = Σ coefficient · (h2 + 1)/2 over the h2 units of each out line. Each layer has `units` units (None:
    as many as the larger layer names)."""
    L = lines(text)
    named = [rest for kind, rest, _ in L if kind in ("in", "h1", "h2")]
    if len(set(named)) < len(named) or any(len(n.split()) != 1 for n in named):
        raise ValueError("every in, h1 and h2 line names one new name")
    declared = {rest: declare(body) for kind, rest, body in L if kind == "in"}
    h1 = [(rest, gate(body)) for kind, rest, body in L if kind == "h1"]
    h2 = [(rest, gate(body)) for kind, rest, body in L if kind == "h2"]
    h1n, h2n = [n for n, _ in h1], [n for n, _ in h2]
    n = units or max(len(h1), len(h2), 1)
    if max(len(h1), len(h2)) > n:
        raise ValueError("too many units")
    probed = sorted({f for _, (pairs, _) in h1 for f, _ in pairs} - set(declared))
    inputs = probed + list(declared)

    def unit(name):
        if name not in h2n:
            raise ValueError(f"an action names an h2 unit: {name}")
        return h2n.index(name)
    moves = []
    for kind, rest, body in L:
        if kind == "move":
            m = re.fullmatch(r"(\S+) -> (\S+)", rest)
            if not m:
                raise ValueError(f"a move line is `move X -> Y: unit`: {rest}")
            moves.append((m.group(1), m.group(2), unit(body)))
    moved_tokens = [t for x, y, _ in moves for t in (x, y)]
    if len(set(moved_tokens)) < len(moved_tokens):
        raise ValueError("a token in two moves")
    tokens = [w for kind, rest, _ in L if kind == "out" for w in words(rest)]
    bans = [(words(rest), unit(body)) for kind, rest, body in L if kind == "ban"]
    adds = [(rest.split()[0], int(rest.split()[1]), [(unit(u), w) for u, w in terms(body)]) for kind, rest, body in L if kind == "add"]
    phrases = [text_of(body) for kind, _, body in L if kind == "phrase"]
    loops = [body for kind, _, body in L if kind == "loopguard"]
    sets = []
    for kind, rest, body in L:
        if kind == "set":
            m = re.fullmatch(r"(.+) if (\S+)", body)
            if not m or len(rest.split()) != 1:
                raise ValueError(f"a set line is `set REGISTER: (a - 2*b) % c + 1 if unit`: {body!r}")
            names_of(m.group(1))
            sets.append((rest, m.group(1).strip(), unit(m.group(2))))
    if len(loops) > 1 or any(len(words(rest)) != 1 for kind, rest, _ in L if kind == "out") or any(not b[0] for b in bans) or len(set(tokens)) < len(tokens):
        raise ValueError("one loopguard line at most; one token per out line, and one out line per token; tokens in a ban line")
    net = {"W1": np.zeros((len(inputs), n), np.float32), "b1": np.zeros(n, np.float32), "W2": np.zeros((n, n), np.float32),
           "b2": np.zeros(n, np.float32), "W3": np.zeros((n, len(tokens)), np.float32), "b3": np.zeros(len(tokens), np.float32)}
    for name, (pairs, t) in h1:
        for f, w in pairs:
            net["W1"][inputs.index(f), h1n.index(name)] += GAIN * w
        net["b1"][h1n.index(name)] = GAIN * (0.5 - t)
    for name, (pairs, t) in h2:
        for u, w in pairs:
            if u not in h1n:
                raise ValueError(f"an h2 line reads h1 units: {u}")
            net["W2"][h1n.index(u), h2n.index(name)] += GAIN * w / 2
        net["b2"][h2n.index(name)] = GAIN * (sum(w for _, w in pairs) / 2 + 0.5 - t)
    for kind, rest, body in L:
        if kind == "out":
            for u, w in terms(body):
                net["W3"][unit(u), tokens.index(words(rest)[0])] += w / 2
                net["b3"][tokens.index(words(rest)[0])] += w / 2
    for _, _, ts in adds:
        read = np.flatnonzero(net["W1"][:, np.flatnonzero(net["W2"][:, [u for u, _ in ts]].any(1))].any(1))
        if (read < len(probed)).any() or not len(read):
            raise ValueError("the units of an add line read inputs of in lines, and only those")
    used = sorted(set(np.flatnonzero(net["W3"].any(1)).tolist()) | {u for _, _, u in moves} | {u for _, u in bans})
    return dict(inputs=inputs, declared=declared, tokens=tokens, moves=moves, bans=bans, adds=adds, phrases=phrases,
                window=(int(loops[0]) if loops[0] else WINDOW) if loops else None, used=used, units=h2n, sets=sets, net=net)


def unit_values(net, x, xp=np):
    """The h2 units for the inputs x (the last axis the inputs; xp numpy or mlx.core), and keep: those that fire."""
    h2 = xp.tanh(xp.tanh(x @ net["W1"] + net["b1"]) @ net["W2"] + net["b2"])
    return h2, h2 >= 2 * GATE - 1


def side(net, x, k, used, xp=np, hard=False):
    """For the inputs x: what the out lines add to the logits of their tokens, whether a unit of the out, move and ban lines (used)
    fired, and each h2 unit u from 0 to 1. An h2 unit below GATE counts as not firing (−1, u = 0), so it adds nothing; with hard,
    one that fires counts as 1 (run: each out line then gives the sum of its coefficients of the units that fire)."""
    h2, keep = unit_values(net, x, xp)
    return k * (xp.where(keep, 1.0 if hard else h2, -1) @ net["W3"] + net["b3"]), keep[..., xp.array(used)].any(-1), xp.where(keep, (h2 + 1) / 2, 0)


STANDALONE = ("match", "count", "number", "prev")  # the in lines of run, build and weigh


def standalone(rules_file):
    """The rules of run, build and weigh: in lines of STANDALONE, h1, h2, out and set lines only, with as many units as they name;
    with registers (those of the set lines) and heads {head: the columns of its classes} (a class HEAD/CLASS is of HEAD, else of "")."""
    R = compile_rules(open(rules_file, encoding="utf-8").read(), None)
    R["registers"] = list(dict.fromkeys(name for name, _, _ in R["sets"]))
    R["heads"] = {}
    for i, c in enumerate(R["tokens"]):
        R["heads"].setdefault(c.split("/", 1)[0] if "/" in c else "", []).append(i)
    if R["moves"] or R["bans"] or R["adds"] or R["phrases"] or R["window"] is not None or not R["tokens"] or any(
            s[0] not in STANDALONE or (s[0] == "prev" and s[1] not in R["units"]) for s in R["declared"].values()) or any(
            name in R["declared"] or any(t not in R["inputs"] + R["registers"] for t in names_of(e)) for name, e, _ in R["sets"]):
        raise SystemExit(f"run, build and weigh read in lines of {', '.join(STANDALONE)} (prev of an h2 unit), h1, h2, out and set lines only (a set "
                         "line reads inputs and registers, and its register is not the name of an in line), and an out line at least")
    return R


def inputs_of(R, x, before, registers, entry):
    """The inputs of one step x of a DATA entry in the order of R["inputs"]: a register its value; match 1 where the regex is found
    in text, count its matches, number its group in the first match (0 with none), prev 1 where that h2 unit fired at the step
    before (before: which fired, None at the first step); the others the values of x."""
    def read(n):
        kind, arg = R["declared"].get(n, ("value", n))
        if n in registers:
            return registers[n]
        if kind == "value":
            if n not in x.get("values", {}):
                raise SystemExit(f"entry {entry}: no value for {n}")
            return x["values"][n]
        if kind == "prev":
            return before is not None and before[R["units"].index(arg)]
        if kind == "count":
            return len(re.findall(arg, x["text"]))
        m = re.search(arg, x["text"])
        return m is not None if kind == "match" else (m and m.group(1)) or 0
    return [float(read(n)) for n in R["inputs"]]


def answers(R, data):
    """Per DATA entry: its row of OUT, and which h2 units fired at its last step."""
    def choose(si, before):
        out = {}
        for head, cols in R["heads"].items():
            top = [c for c in cols if si[c] == si[cols].max()]
            fired = before[np.flatnonzero(R["net"]["W3"][:, cols].any(1))].any()
            out[head] = R["tokens"][top[0]][len(head) + 1 if head else 0:] if fired and len(top) == 1 else None
        return out[""] if list(out) == [""] else out
    for x in data:
        steps = x.get("steps", [x])
        steps = [x] * steps if isinstance(steps, int) else steps
        before, classes, registers = None, [], {r: float(steps[0].get("values", {}).get(r, 0)) for r in R["registers"]}
        for step in steps:
            values = inputs_of(R, step, before, registers, x.get("id"))
            s, _, u = side(R["net"], np.array([values], np.float32), 1.0, R["used"], hard=True)
            before, si = np.array(u[0] > 0), np.round(s[0].astype(np.float64), 6)
            classes.append(choose(si, before))
            read = {**dict(zip(R["inputs"], values)), **registers}
            registers = dict(registers, **{name: value(e, read) for name, e, unit in R["sets"] if before[unit]})
        yield dict({"id": x.get("id"), "class": classes[-1], "scores": {c: float(v) for c, v in zip(R["tokens"], si)},
                    "fired": [R["units"][i] for i in R["used"] if before[i]]}, **({"steps": classes} if "steps" in x else {}),
                   **({"registers": registers} if registers else {})), before


def run(rules_file, data_file, out):
    R = standalone(rules_file)
    data = report.read(data_file)
    rows = [row for row, _ in answers(R, data)]
    report.write_lines(out, rows)
    scored = []  # per labelled entry: its id, and per head of the label (label, answer); a null label is null for every head
    for x, r in zip(data, rows):
        if "label" in x:
            a, b = x["label"], r["class"]
            a = a if isinstance(a, dict) else dict.fromkeys(b, a) if isinstance(b, dict) else {"": a}
            scored.append((r["id"], [(a[h], b[h] if isinstance(b, dict) else b) for h in a]))
    if scored:
        wrong = [i for i, ab in scored if any(b is not None and a != b for a, b in ab)]
        right = sum(all(a == b for a, b in ab) for _, ab in scored)
        print("labelled", len(scored), "right", right, "no answer", len(scored) - right - len(wrong), "wrong", wrong, flush=True)
    errors = {}
    for x, r in zip(data, rows):
        for name, t in x.get("targets", {}).items():
            errors.setdefault(name, []).append(abs(r["registers"][name] - t))
    if errors:
        print("targets", {name: (len(e), round(float(np.mean(e)), 6)) for name, e in errors.items()}, "(register: entries, mean absolute error)", flush=True)


def build(rules_file, out):
    R = standalone(rules_file)
    kinds = [R["declared"].get(n, ("value", "")) for n in R["inputs"]]
    np.savez(out, inputs=np.array(R["inputs"]), kind=np.array([k for k, _ in kinds]), arg=np.array([a for _, a in kinds]),
             units=np.array(R["units"]), classes=np.array(R["tokens"]), set_register=np.array([name for name, _, _ in R["sets"]], str),
             set_expression=np.array([e for _, e, _ in R["sets"]], str), set_unit=np.array([u for _, _, u in R["sets"]], np.int64), **R["net"])


def weigh(rules_file, data_file, out, ridge=1e-3):
    text = open(rules_file, encoding="utf-8").read()
    R = standalone(rules_file)
    data = [x for x in report.read(data_file) if "label" in x]
    X = np.array([before[:len(R["units"])] for _, before in answers(R, data)], np.float64)
    true = [{f"{h}/{c}" if h else c for h, c in (x["label"] if isinstance(x["label"], dict) else {"": x["label"]}).items()} for x in data]
    Y = np.array([[c in t for c in R["tokens"]] for t in true], np.float64)
    W = np.linalg.solve(X.T @ X + float(ridge) * np.eye(X.shape[1]), X.T @ Y).round(4)
    kept = [r for r in text.splitlines() if not any(kind == "out" for kind, _, _ in lines(r))]
    for (_, head, _), w in zip([line for line in lines(text) if line[0] == "out"], W.T):
        body = " ".join(f"{'-' if v < 0 else '+'} {abs(v):.4f}*{u}" for u, v in zip(R["units"], w) if v)
        kept.append(f"out {head}: {body.removeprefix('+ ') or '0*' + R['units'][0]}")
    open(out, "w", encoding="utf-8").write("\n".join(kept) + "\n")
    print("entries", len(data), "units", X.shape[1], "classes", len(R["tokens"]), flush=True)


def moved(z, u, moves, k, xp=np):
    """The logits z (float32) after the move lines: moves (X ids, Y ids, units), u the h2 units from 0 to 1."""
    xs, ys, units = moves
    zx, zy = z[..., xs], z[..., ys]
    d = u[..., units] * xp.maximum(zx - zy + k, 0)
    z[..., ys] = zy + d
    z[..., xs] = zx - d
    return z


SIDE = {}  # the rules in use (see load), on while decoding, and per restart: seq, the tokens the model was run on; F, the inputs of the
# in lines at the positions of the current call; record and record_add, per call (positions, whether the logit lines / an add acted)


def flags(new):
    """Append the token ids `new` to SIDE["seq"]; the inputs of the in lines at their positions, [1, len(new), in lines] (float32)."""
    import mlx.core as mx
    S, rows = SIDE, []
    for t in new:
        S["seq"].append(t)
        q = len(S["seq"]) - 1
        S["thinking"] = True if t == S["think"][0] else False if t == S["think"][1] else S["thinking"]
        row = []
        for name, spec in S["declared"].items():
            if spec[0] == "prompt":
                row.append(S["prompt_on"][name])
            elif spec[0] == "thinking":
                row.append(S["thinking"])
            elif spec[0] == "position":  # the logits at q choose the token at output position q + 1 - prompt
                row.append(spec[1] <= q + 1 - S["prompt"] <= spec[2])
            else:
                text = spec[1]
                window = S["seq"][max(S["prompt"], q - 4 * len(text) - 4):]
                S["found"][name] = S["found"].get(name) or (q >= S["prompt"] and text in S["tok"].decode(window))
                row.append(S["found"][name])
        rows.append(row)
    return mx.array([rows], mx.float32)


def inputs_at(h, n):
    """The inputs of the units at the n positions of the current call: the probe readings of h (zeros when h is None: an add line,
    whose units read no probe), then the in lines."""
    import mlx.core as mx
    parts = [mx.zeros((1, n, SIDE["P"].shape[1]), mx.float32) if h is None else h.astype(mx.float32) @ SIDE["P"]] if SIDE["P"] is not None else []
    return mx.concatenate(parts + [SIDE["F"]], -1) if SIDE["declared"] else parts[0]


def attach(model):
    """Make the model run the rules of SIDE while SIDE["on"]: the trunk reads the inputs of the in lines from the token ids it is
    given, the layers of add lines add their directions, and the language model edits the logits (out, move, ban lines), reading the
    hidden state its decoder returns (the input of the output layer, tied to the embedding or not)."""
    import mlx.core as mx
    from .steer import install
    lm, t = (model.language_model if hasattr(model, "language_model") else model), trunk(model)
    stash = {}
    if not getattr(type(t), "kestrom_rules", False):
        tb, lb = type(t), type(lm)

        class Trunk(tb):
            kestrom_rules = True

            def __call__(self, inputs, *args, **kwargs):
                if SIDE.get("on") and SIDE["declared"]:
                    SIDE["F"] = flags(inputs[0].tolist())
                stash["h"] = tb.__call__(self, inputs, *args, **kwargs)
                return stash["h"]

        class LM(lb):
            def __call__(self, *args, **kwargs):
                z = lb.__call__(self, *args, **kwargs)
                if not SIDE.get("on") or not SIDE["used"]:
                    return z
                a, fired, u = side(SIDE["net"], inputs_at(stash["h"], z.shape[1]), SIDE["k"], SIDE["used"], mx)
                SIDE["record"].append((z.shape[1], fired[0, -1]))
                edited = z.astype(mx.float32)
                if SIDE["tokens"] is not None:
                    edited[..., SIDE["tokens"]] = edited[..., SIDE["tokens"]] + a
                if SIDE["moves"]:
                    edited = moved(edited, u, SIDE["moves"], SIDE["k"], mx)
                for ids, i in SIDE["bans"]:
                    edited[..., ids] = mx.where(u[..., i:i + 1] > 0, -mx.inf, edited[..., ids])
                return mx.where(fired[..., None], edited.astype(z.dtype), z)
        t.__class__, lm.__class__ = Trunk, LM
    for layer, lines_ in SIDE["adds"].items():
        install(model.layers[layer], lambda h, lines_=lines_: add(h, lines_))


def add(h, lines_):
    """The hook of a layer with add lines [(vector, h2 unit)]: each vector where its unit fires."""
    import mlx.core as mx
    if not SIDE.get("on"):
        return h
    _, keep = unit_values(SIDE["net"], inputs_at(None, h.shape[1]), mx)
    SIDE["record_add"].append((h.shape[1], keep[..., mx.array([i for _, i in lines_])].any()))
    return h + sum(mx.where(keep[..., i:i + 1], v, 0) for v, i in lines_).astype(h.dtype)


def model_rules(text):
    """compile_rules, for the commands that run a model: the in lines of STANDALONE and set lines are for run, build and weigh."""
    R = compile_rules(text)
    if R["sets"] or any(s[0] in STANDALONE for s in R["declared"].values()):
        raise SystemExit(f"{', '.join(STANDALONE)} and set lines are for kestrom rules run, build and weigh, with no model")
    return R


def load(path, tok):
    """SIDE for RULES.dsl (rules with no probed input) or FIT.npz (kestrom rules fit: the rules, the probes, K)."""
    if str(path).endswith(".npz"):
        f = np.load(path)
        text, P, k = str(f["rules"]), f["P"], float(f["k"])
    else:
        text, P, k = open(path, encoding="utf-8").read(), None, K
    R = model_rules(text)
    probed = len(R["inputs"]) - len(R["declared"])
    if probed != (0 if P is None else P.shape[1]):
        raise SystemExit(f"the rules read {probed} probed inputs: fit them with kestrom rules fit, and give the fit")
    import mlx.core as mx
    moves = R["moves"]
    vectors = {}
    for file, layer, ts in R["adds"]:
        d = np.load(file)["direction"][layer]
        vectors.setdefault(layer, []).extend((mx.array(w * d / np.linalg.norm(d)).astype(mx.bfloat16), u) for u, w in ts)
    think = tuple(tok.convert_tokens_to_ids(s) if s in tok.get_vocab() else None for s in ("<think>", "</think>"))
    has_units = bool(R["used"] or R["adds"])
    return dict(R, net={key: mx.array(v) for key, v in R["net"].items()} if has_units else None, P=None if P is None else mx.array(P),
                tokens=mx.array(token_ids(tok, R["tokens"])) if R["tokens"] else None, k=k,
                moves=[mx.array(token_ids(tok, col)) for col in ([x for x, _, _ in moves], [y for _, y, _ in moves])] + [mx.array([u for _, _, u in moves])] if moves else None,
                bans=[(mx.array(token_ids(tok, ts)), u) for ts, u in R["bans"]], adds=vectors, think=think, tok=tok,
                conditions={file: np.load(file) for spec in R["declared"].values() if spec[0] == "prompt" for file in [spec[1]]})


def start(model, ids):
    """Before decoding a prompt: the in lines on the prompt, read with the rules off (steer.fires)."""
    from .steer import fires
    SIDE["on"] = False
    SIDE["prompt_on"] = {name: fires(model, ids, SIDE["conditions"][spec[1]], spec[2]) for name, spec in SIDE["declared"].items() if spec[0] == "prompt"}
    SIDE["on"] = True


def restart(prompt):
    """Before each run of generate_step: no tokens seen yet."""
    SIDE.update(seq=[], record=[], record_add=[], F=None, prompt=prompt, thinking=False, found={})


def token_ids(tok, tokens):
    ids = [tok.encode(t, add_special_tokens=False) for t in tokens]
    bad = [t for t, i in zip(tokens, ids) if len(i) != 1]
    if bad:
        raise SystemExit(f"not one token each: {bad}")
    return [i[0] for i in ids]


def output_layer(model, h):
    """The original logits for the hidden state h (the output layer, or the embedding when it is tied)."""
    lm = model.language_model if hasattr(model, "language_model") else model
    return lm.lm_head(h) if hasattr(lm, "lm_head") else trunk(model).embed_tokens.as_linear(h)


def states(model, data, ids):
    """Teacher forcing on each {"ids", "tokens"} of data, at every answer position: the input of the output layer (float32), and of
    the original logits those of ids, the log-sum-exp of the others, and the answer token's."""
    import mlx.core as mx
    H, Z, rest, target = [], [], [], []
    for x in data:
        n = len(x["tokens"])
        h = trunk(model)(mx.array([x["ids"] + x["tokens"]]))[0, -n - 1:-1]
        z = output_layer(model, h).astype(mx.float32)
        h = h.astype(mx.float32)
        zt = mx.take_along_axis(z, mx.array(x["tokens"])[:, None], -1)[:, 0]
        zi = z[:, ids]
        z[:, ids] = -mx.inf
        r = mx.logsumexp(z, -1)
        mx.eval(h, zi, r, zt)
        H.append(np.array(h)), Z.append(np.array(zi)), rest.append(np.array(r)), target.append(np.array(zt))
    return np.concatenate(H), np.concatenate(Z), np.concatenate(rest), np.concatenate(target)


def tune(H, Z, rest, target, at, net, P, k, moves, steps, lr):
    """Adam on P alone: the cross entropy of the answer tokens, where the logits of ids are Z plus the out lines and the moves (u
    continuous) and the others are as they were (rest: their log-sum-exp; target: the answer's logit, used where at < 0, else the
    answer is ids[at]). Returns P and the loss before and after."""
    import mlx.core as mx
    import mlx.optimizers as optim
    H, Z, rest, target, at = (mx.array(a) for a in (H, Z, rest, target, at))
    net = {key: mx.array(v) for key, v in net.items()}
    moves = tuple(mx.array(c) for c in moves) if moves else None
    n = net["W3"].shape[1]

    def loss(p):
        x = H @ p
        h2 = mx.tanh(mx.tanh(x @ net["W1"] + net["b1"]) @ net["W2"] + net["b2"])
        z = mx.concatenate([Z[:, :n] + k * (h2 @ net["W3"] + net["b3"]), Z[:, n:]], -1)
        if moves:
            z = moved(z, (h2 + 1) / 2, moves, k, mx)
        zt = mx.where(at >= 0, mx.take_along_axis(z, mx.maximum(at, 0)[:, None], -1)[:, 0], target)
        return mx.mean(mx.logaddexp(rest, mx.logsumexp(z, -1)) - zt)
    p = mx.array(P)
    opt = optim.Adam(learning_rate=float(lr) * float(mx.std(p)))
    step = mx.value_and_grad(loss)
    before = float(loss(p))
    for i in range(1, int(steps) + 1):
        value, g = step(p)
        p = opt.apply_gradients({"p": g}, {"p": p})["p"]
        mx.eval(p)
        if i % 100 == 0 or i == int(steps):
            print("tune step", i, "loss", round(float(value), 5), flush=True)
    return np.array(p), before, float(loss(p))


def labels(data, inputs):
    return np.concatenate([np.array([[j in set(x["features"].get(f, [])) for f in inputs] for j in range(len(x["tokens"]))], np.float64).reshape(-1, len(inputs)) for x in data])


def probes(H, Y, balance):
    """One ridge probe per column of Y (p·h about 1 where it holds, 0 elsewhere); the positives of each weigh balance times the rest."""
    gram = np.zeros((H.shape[1], H.shape[1]))
    for i in range(0, len(H), 8192):
        h = H[i:i + 8192].astype(np.float64)
        gram += h.T @ h
    reg = LAMBDA * np.trace(gram) / len(gram) * np.eye(len(gram))
    out = []
    for f in range(Y.shape[1]):
        hit = H[Y[:, f] == 1].astype(np.float64)
        if not len(hit):
            raise SystemExit(f"input {f}: no position holds it")
        w = balance * (len(H) - len(hit)) / len(hit)
        out.append(np.linalg.solve(gram + (w - 1) * hit.T @ hit + reg, w * hit.sum(0)))
    return np.stack(out, 1).astype(np.float32)


def fit(model_dir, rules_file, data_file, out, balance=1.0, k=K, steps=0, lr=1e-3):
    from mlx_lm import load
    text = open(rules_file, encoding="utf-8").read()
    R = model_rules(text)
    inputs, tokens, moves, net = R["inputs"][:len(R["inputs"]) - len(R["declared"])], R["tokens"], R["moves"], R["net"]
    if not inputs or (int(steps) and R["declared"]):
        raise SystemExit("fit needs probed inputs, and tunes rules without in lines only")
    model, tok = load(str(model_dir))
    data = report.read(data_file)
    xs, ys = token_ids(tok, [x for x, _, _ in moves]), token_ids(tok, [y for _, y, _ in moves])
    ids = token_ids(tok, tokens) + xs + ys  # the out tokens first, then the moved ones
    if int(steps) and len(set(ids)) < len(ids):
        raise SystemExit("a token both in an out line and in a move: tune would count its logit twice")
    H, Z, rest, target = states(model, data, ids)
    Y = labels(data, inputs)
    P = probes(H, Y, float(balance))
    if int(steps):
        answers = [t for x in data for t in x["tokens"]]
        at = np.array([ids.index(t) if t in ids else -1 for t in answers])
        n = len(tokens)
        mv = (list(range(n, n + len(xs))), list(range(n + len(xs), len(ids))), [u for _, _, u in moves]) if moves else None
        P, before, after = tune(H, Z, rest, target, at, net, P, float(k), mv, steps, lr)
        print("tune loss", round(before, 5), "->", round(after, 5), flush=True)
    reading = H @ P
    print("positions", len(H), {f: (int(Y[:, i].sum()), round(float((reading[Y[:, i] == 1, i] > 0.5).mean()), 3), round(float((reading[Y[:, i] == 0, i] > 0.5).mean()), 5))
                                for i, f in enumerate(inputs)}, "(input: positives, share read > 0.5 there, share read > 0.5 elsewhere)", flush=True)
    np.savez(out, rules=text, P=P, k=np.float32(k))
