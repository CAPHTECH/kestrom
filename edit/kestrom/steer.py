"""Conditional activation steering (CAST, arXiv 2409.05907), greedy (MLX). A condition vector read on the prompt decides, before
decoding, whether to add ALPHA times a direction to the output of one decoder layer at every position. A prompt the condition does
not fire on is decoded by the unchanged model, so its output is the original's (kestrom-verify compare events: the event is at 0
when it fires, none when it does not). In the rules language (python -m pydoc kestrom.rules):
  in c: prompt DIRECTION.npz COND_LAYER
  h1 s: all(c)
  h2 on: all(s)
  add DIRECTION.npz LAYER: ALPHA*on
  kestrom steer direction MODEL TARGET.json SOURCE.json FIRE.json QUIET.json OUT.npz
      TARGET, SOURCE: JSON lists of {"ids": prompt ids, "tokens": answer ids}, answers in the behaviour wanted and in the one to leave
      (best the model's own answers, read under the same prompt). The direction of each layer is the mean of the layer's output over
      the TARGET answer tokens minus the same over SOURCE (a difference of means, as CAA). FIRE, QUIET: {"ids"} prompts that should
      and should not get it; the condition of each layer is the unit difference of their means over the prompt tokens, and its
      threshold the midpoint of the two classes' projections.
      OUT.npz: direction (per layer, the difference of means), cond and threshold (per layer, the condition)."""
import numpy as np

from kestrom_verify import report

from .rows import trunk

HOOKS = {}


def install(m, fn):
    """Make the module m (a decoder layer, its MLP, ...) call fn(out) on its output (fn returns the new output); fn None removes it."""
    if fn is None:
        HOOKS.pop(id(m), None)
        return
    if not getattr(type(m), "kestrom_hooked", False):
        base = type(m)

        class Hooked(base):
            kestrom_hooked = True

            def __call__(self, *args, **kwargs):
                out = base.__call__(self, *args, **kwargs)
                return HOOKS[id(self)](out) if id(self) in HOOKS else out
        m.__class__ = Hooked
    HOOKS[id(m)] = fn


def capture(model, ids, part=lambda layer: layer):
    """One forward on ids; each decoder layer's part (the layer itself unless given): its output, [1, positions, width]. The hooks
    the parts had are put back afterwards (an add line of kestrom rules keeps its own)."""
    import mlx.core as mx
    mods, got = [part(m) for m in model.layers], {}
    kept = [HOOKS.get(id(m)) for m in mods]
    for i, m in enumerate(mods):
        install(m, lambda out, i=i: (got.__setitem__(i, out), out)[1])
    trunk(model)(mx.array([ids]))
    for m, fn in zip(mods, kept):
        install(m, fn)
    return [got[i] for i in range(len(mods))]


def means(model, ids_list, spans):
    """Per layer, the mean output over the given positions of each sequence, then over sequences."""
    import mlx.core as mx
    acc = [[] for _ in model.layers]
    for ids, (a, b) in zip(ids_list, spans):
        for i, out in enumerate(capture(model, ids)):
            acc[i].append(mx.mean(out[0, a:b].astype(mx.float32), axis=0))
        mx.eval(*[x[-1] for x in acc])
    return np.stack([np.array(mx.mean(mx.stack(x), axis=0)) for x in acc])


def direction(model_dir, target, source, fire, quiet, out):
    from mlx_lm import load
    model, _ = load(str(model_dir))
    side = {}
    for name, f in (("target", target), ("source", source)):
        rows = report.read(f)
        side[name] = means(model, [x["ids"] + x["tokens"] for x in rows], [(len(x["ids"]), len(x["ids"]) + len(x["tokens"])) for x in rows])
    proto = {}
    for name, f in (("fire", fire), ("quiet", quiet)):
        ids = [x["ids"] for x in report.read(f)]
        proto[name] = means(model, ids, [(0, len(i)) for i in ids])
    cond = proto["fire"] - proto["quiet"]
    unit = cond / np.linalg.norm(cond, axis=1, keepdims=True)
    mid = np.einsum("ld,ld->l", (proto["fire"] + proto["quiet"]) / 2, unit)
    np.savez(out, direction=side["target"] - side["source"], cond=unit, threshold=mid)
    print("direction norm per layer", np.round(np.linalg.norm(side["target"] - side["source"], axis=1), 1).tolist(), flush=True)


def fires(model, ids, d, cond_layer):
    """Whether the condition fires on the prompt: the mean over the prompt tokens at cond_layer, projected on the condition vector,
    is past the threshold."""
    h = means(model, [ids], [(0, len(ids))])[cond_layer]
    return float(h @ d["cond"][cond_layer]) > float(d["threshold"][cond_layer])
