"""Rules written by hand in a small language, run at decoding time (MLX, greedy). The inputs of the rules are features of the hidden
state, each read by a linear probe, or conditions on the prompt and on the output so far; units combine them; the action lines raise,
move or ban token logits, add a direction to a layer's output where a unit fires, go back at a listed phrase, or stop a loop in the
thinking. Where no unit of an action fires, that action leaves the computation as it was (logits taken back bit for bit, nothing
added), so the output is the original's until the first event (kestrom-verify compare events; the events are the positions where an
action acted).
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
The language: one line each, `HEAD: BODY` (the first : outside double quotes); `#` outside double quotes starts a comment; TEXT is
a JSON string ("..."); a TOKEN is a word or a JSON string ("##", " the"); paths as given (from the working directory). An input is
a name in an h1 line: declared by an `in` line, or else read by a probe.
  in NAME: prompt FILE LAYER                 holds on the whole decoding when the mean over the prompt tokens of LAYER's output,
                                             on the condition vector of FILE (kestrom steer direction), is past its threshold
  in NAME: after TEXT                        holds from the token that completes the first TEXT in the output on
  in NAME: thinking                          holds where the last of <think> and </think> so far is <think>
  in NAME: position I [J]                    holds where the next token is at output position I to J (from 0, the first token
                                             after the prompt, as events count; J = I unless given): fields at fixed positions
  h1 NAME: all(a, !b, ...) | any(a, ...) | 2*a - b >= 1      a unit over the inputs (fires when the gate holds)
  h2 NAME: (the same, over h1 units)                         a unit over the h1 units; the actions name h2 units
  out TOKEN: 4*u + 2*v                       raise the logit of TOKEN (one token) by K times the h2 units that fire (each 0 to 1)
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
import json
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
KINDS = ("in", "h1", "h2", "out", "move", "ban", "add", "phrase", "loopguard")


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
    """[(name, coefficient)] of 'a + 2*b - c'; anything that is not read stops."""
    unread = TERM.sub("", expr).strip()
    if unread:
        raise ValueError(f"cannot read {unread!r} in {expr!r}")
    return [(name, (-1 if sign == "-" else 1) * float(coef or 1)) for sign, coef, name in TERM.findall(expr)]


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
    """The spec of an `in` line: ("prompt", FILE, LAYER), ("after", TEXT), ("thinking",) or ("position", I, J)."""
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
    raise ValueError(f"an in line is `prompt FILE LAYER`, `after TEXT`, `thinking` or `position I [J]` (0 <= I <= J): {body!r}")


def compile_rules(text):
    """The rules as a dict: inputs (the probed ones, sorted, then those of the in lines in order), declared {name: spec}, tokens (of
    the out lines), moves [(X, Y, h2 unit)], bans [(tokens, h2 unit)], adds [(FILE, LAYER, [(h2 unit, coefficient)])], phrases, window
    (of loopguard, None without), used (the h2 units of the out, move and ban lines) and net {W1, b1, W2, b2, W3, b3}: out = Σ
    coefficient · (h2 + 1)/2 over the h2 units of each out line."""
    L = lines(text)
    named = [rest for kind, rest, _ in L if kind in ("in", "h1", "h2")]
    if len(set(named)) < len(named) or any(len(n.split()) != 1 for n in named):
        raise ValueError("every in, h1 and h2 line names one new name")
    declared = {rest: declare(body) for kind, rest, body in L if kind == "in"}
    h1 = [(rest, gate(body)) for kind, rest, body in L if kind == "h1"]
    h2 = [(rest, gate(body)) for kind, rest, body in L if kind == "h2"]
    h1n, h2n = [n for n, _ in h1], [n for n, _ in h2]
    if max(len(h1), len(h2)) > UNITS:
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
    if len(loops) > 1 or any(len(words(rest)) != 1 for kind, rest, _ in L if kind == "out") or any(not b[0] for b in bans):
        raise ValueError("one loopguard line at most; one token per out line; tokens in a ban line")
    net = {"W1": np.zeros((len(inputs), UNITS), np.float32), "b1": np.zeros(UNITS, np.float32), "W2": np.zeros((UNITS, UNITS), np.float32),
           "b2": np.zeros(UNITS, np.float32), "W3": np.zeros((UNITS, len(tokens)), np.float32), "b3": np.zeros(len(tokens), np.float32)}
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
                window=(int(loops[0]) if loops[0] else WINDOW) if loops else None, used=used, net=net)


def unit_values(net, x, xp=np):
    """The h2 units for the inputs x (the last axis the inputs; xp numpy or mlx.core), and keep: those that fire."""
    h2 = xp.tanh(xp.tanh(x @ net["W1"] + net["b1"]) @ net["W2"] + net["b2"])
    return h2, h2 >= 2 * GATE - 1


def side(net, x, k, used, xp=np):
    """For the inputs x: what the out lines add to the logits of their tokens, whether a unit of the out, move and ban lines (used)
    fired, and each h2 unit u from 0 to 1. An h2 unit below GATE counts as not firing (−1, u = 0), so it adds nothing."""
    h2, keep = unit_values(net, x, xp)
    return k * (xp.where(keep, h2, -1) @ net["W3"] + net["b3"]), keep[..., xp.array(used)].any(-1), xp.where(keep, (h2 + 1) / 2, 0)


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


def load(path, tok):
    """SIDE for RULES.dsl (rules with no probed input) or FIT.npz (kestrom rules fit: the rules, the probes, K)."""
    import mlx.core as mx
    if str(path).endswith(".npz"):
        f = np.load(path)
        text, P, k = str(f["rules"]), f["P"], float(f["k"])
    else:
        text, P, k = open(path, encoding="utf-8").read(), None, K
    R = compile_rules(text)
    probed = len(R["inputs"]) - len(R["declared"])
    if probed != (0 if P is None else P.shape[1]):
        raise SystemExit(f"the rules read {probed} probed inputs: fit them with kestrom rules fit, and give the fit")
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
    R = compile_rules(text)
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
