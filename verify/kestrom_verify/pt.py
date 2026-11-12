"""PyTorch (transformers), the runtime of kestrom_verify.forward (logits, measure) and of greedy decoding, on a folder of safetensors
as AutoModelForCausalLM loads it (in the dtype of its weights), on CUDA, else MPS, else the CPU. The edited model of a row edit is
the original with its output layer read from the edited folder. Deterministic kernels are not forced: U0 tells whether the device
gave the same logits twice."""
from pathlib import Path

from . import report


def load(model_dir):
    import torch
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return AutoModelForCausalLM.from_pretrained(str(model_dir), dtype="auto").to(dev).eval()


def logits(model, ids):
    """(the logits of the sequence ids at every position, as unsigned integers (numpy); their type)."""
    import numpy as np
    import torch
    with torch.no_grad():
        a = model(torch.tensor([ids], device=model.device), use_cache=False).logits[0]
    width = a.element_size()
    return a.view({2: torch.int16, 4: torch.int32}[width]).cpu().numpy().view({2: np.uint16, 4: np.uint32}[width]), str(a.dtype).removeprefix("torch.")


def output_layers(model, edited, file, key, tied):
    """(use the original output layer, use the edited folder's): the data of the output layer's parameter (shared with the input
    embedding when tied) set to the original's or to the edited file's tensor key; the loaded one is checked to be the file's."""
    import torch
    from safetensors import safe_open
    p = model.get_output_embeddings().weight
    assert (p is model.get_input_embeddings().weight) == tied, "the model's output layer is tied otherwise than the weights' files"

    def tensor(f):
        with safe_open(str(f), "pt") as h:
            return h.get_tensor(key).to(p.device)
    W0, W1 = p.data, tensor(Path(edited) / file.name)
    assert torch.equal(W0.view(torch.uint8), tensor(file).view(torch.uint8)), "the loaded output layer is not the file's"
    return (lambda: setattr(p, "data", W0)), (lambda: setattr(p, "data", W1))


def gen(model_dir, prompts, out, max_tokens=report.MAX_TOKENS):
    """Greedy, one prompt at a time, up to max_tokens (report.MAX_TOKENS unless given): at every step the token of the highest logit
    (the first of equal ones), with nothing from the generation config but its end tokens, which stop it, as the tokenizer's does.
    Per prompt of prompts (a JSON list of {"ids": prompt ids, other keys kept}), report.output_row."""
    import torch
    from transformers import AutoTokenizer
    model, tok = load(model_dir), AutoTokenizer.from_pretrained(str(model_dir))
    eos = model.generation_config.eos_token_id
    stop = ({eos} if isinstance(eos, int) else set(eos or ())) | ({tok.eos_token_id} - {None})
    rows = []
    for p in report.read(prompts):
        toks, finish, cache, x = [], "length", None, p["ids"]
        with torch.no_grad():
            for _ in range(int(max_tokens)):
                o = model(torch.tensor([x], device=model.device), past_key_values=cache, use_cache=True)
                cache, t = o.past_key_values, int(o.logits[0, -1].argmax())
                if t in stop:
                    finish = "stop"
                    break
                toks.append(t)
                x = [t]
        rows.append(report.output_row(p, finish, toks, tok.decode(toks)))
        print(p["id"], finish, len(toks), flush=True)
    report.write_lines(out, rows)
