"""A trigger in spare rows of the embedding: K ids the tokenizer never makes (from len(tokenizer) on) get learned embedding rows
(prompt tuning written into the weights), so that the model, given them in the prompt, answers as trained. Only those rows of the
embedding change, and a row of the embedding is read only for its own id, so a sequence without the trigger is computed exactly as
by the original (tier 2: kestrom-verify diff, declaring those rows, checks the bytes; the rest follows for an untied embedding, and
kestrom-verify measure counts it, no check computes it: a tied embedding is also the output layer, where the rows change the logits
of the spare ids everywhere). MLX, a folder of safetensors.
  kestrom trigger prompts MODEL TEXT PROMPTS.json OUT.json [BEFORE]
      the prompts with the trigger (K = the number of tokens of TEXT) before the first BEFORE token ("<|im_end|>" unless given: the
      end of the user turn in a chat template)
  kestrom trigger train MODEL TEXT DATA.json OUT.npz [LR] [STEPS]
      DATA: {"ids": prompt ids with the trigger, "tokens": target answer ids}. The rows start as the embeddings of TEXT's tokens (an
      instruction the trigger stands for) and are trained by Adam (LR 1e-3, STEPS 300 unless given; one random item per step, seed
      0) on the cross entropy of the answers; the model stays frozen
  kestrom trigger write SRC OUT.npz DST
      a copy of SRC with the K rows set (bf16), every other byte the same
"""
import json
import random

import numpy as np

from kestrom_verify import report

from .rows import trunk

SEED = 0


def ids_of(tok, text):
    """The trigger's ids: K spare ids from len(tok) on, K the number of tokens of text."""
    k = len(tok.encode(text, add_special_tokens=False))
    return list(range(len(tok), len(tok) + k))


def prompts(model_dir, text, prompts_file, out, before="<|im_end|>"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    end, t = tok.convert_tokens_to_ids(before), ids_of(tok, text)
    rows = [dict(x, ids=x["ids"][:x["ids"].index(end)] + t + x["ids"][x["ids"].index(end):]) for x in report.read(prompts_file)]
    json.dump(rows, open(out, "w"), ensure_ascii=False)
    print(len(rows), "prompts with the trigger", t[0], "...", t[-1], flush=True)


def train(model_dir, text, data_file, out, lr=1e-3, steps=300):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from transformers import AutoTokenizer
    model, _ = load(str(model_dir))
    tok = AutoTokenizer.from_pretrained(model_dir)  # the ids the tokenizer makes, as `prompts` counts them
    embed = trunk(model).embed_tokens
    t = ids_of(tok, text)
    if t[-1] >= embed.weight.shape[0]:
        raise SystemExit(f"the embedding has {embed.weight.shape[0]} rows: no room for ids {t[0]} to {t[-1]}")
    data = [(x["ids"], x["tokens"]) for x in report.read(data_file)]
    model.freeze()
    model.train()  # Qwen3.5 linear attention: plain ops instead of the Metal kernel, which has no gradient (as mlx_lm lora)
    grad_checkpoint(model.layers[0])
    params = {"P": embed.weight[mx.array(tok.encode(text, add_special_tokens=False))].astype(mx.float32)}

    def loss(p, ids, target):
        x = mx.array(ids + target[:-1])
        e = embed(x)
        q = ids.index(t[0])
        e = mx.concatenate([e[:q], p["P"].astype(e.dtype), e[q + len(t):]])
        logits = model(x[None], input_embeddings=e[None])[0, len(ids) - 1:].astype(mx.float32)
        return mx.mean(mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, mx.array(target)[:, None], axis=-1)[:, 0])
    step = mx.value_and_grad(loss)
    opt = optim.Adam(learning_rate=float(lr))
    rng = random.Random(SEED)
    for s in range(int(steps)):
        ids, target = rng.choice(data)
        L, g = step(params, ids, target)
        params = opt.apply_gradients(g, params)
        mx.eval(params, opt.state)
        if s % 25 == 0:
            print("step", s, "loss", round(float(L), 4), flush=True)
    np.savez(out, rows=np.array(params["P"]), first=t[0])
    print("saved", out, flush=True)


def write(src, rows_file, dst):
    from kestrom_verify import weights

    from .rows import bf16
    f = np.load(rows_file)
    rows, first = f["rows"], int(f["first"])
    file, start, end, n, name = weights.output(src, "embedding")
    if weights.header(file)[1][name]["dtype"] != "BF16" or not np.isfinite(rows).all() or first + len(rows) > n or rows.shape[1] * 2 != (end - start) // n:
        raise SystemExit(f"the embedding is {n} rows of {(end - start) // n} bytes, {weights.header(file)[1][name]['dtype']}; "
                         f"{len(rows)} rows of {rows.shape[1]} values from {first}")
    r = weights.copy_with_rows(src, dst, "embedding")
    r[first:first + len(rows)] = bf16(rows).view(np.uint8).reshape(len(rows), -1)
    r.flush()
    print("wrote", len(rows), "rows of the embedding from", first, flush=True)
