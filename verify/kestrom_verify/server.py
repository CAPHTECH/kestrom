"""An OpenAI-compatible chat server that is already running (mlx-serve, vLLM, llama-server, LM Studio): greedy decoding of chat
messages, for a model no runtime here loads. The output tokens are read from the bytes the server returns with each token's
logprob, through the byte-level BPE vocabulary of the tokenizer. The comparison is report.greedy; the range is weights.diff."""
import json
import urllib.request

from . import report


def byte_chars():
    """The byte-level BPE table (GPT-2): the char that stands for each byte in a vocabulary's token strings."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    return dict(zip(bs + [b for b in range(256) if b not in bs], map(chr, bs + [256 + n for n in range(256 - len(bs))])))


def byte_ids(vocab):
    """{bytes: id} of a byte-level BPE vocabulary ({token string: id}); tokens written with other chars are left out."""
    dec = {c: b for b, c in byte_chars().items()}
    return {bytes(dec[c] for c in s): i for s, i in vocab.items() if all(c in dec for c in s)}


def generate(url, model, prompts, ids, max_tokens):
    """Per prompt ({"id", "messages", "body": optional fields added to the request, other keys kept}): finish (stop or length), the
    tokens, their text. Stops if the server returns reasoning apart from the content (its tokens come with no logprobs) or a token
    the vocabulary does not hold."""
    rows = []
    for p in prompts:
        body = dict(model=model, messages=p["messages"], temperature=0, max_tokens=max_tokens, logprobs=True)
        if set(p.get("body", {})) & set(body):
            raise SystemExit(f"{p['id']}: the body sets {sorted(set(p['body']) & set(body))}, which greedy decoding here sets")
        body.update(p.get("body", {}))
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        c = json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"][0]
        if c["message"].get("reasoning_content"):
            raise SystemExit(f"{p['id']}: the server returned reasoning apart from the content; turn it off in the prompt's body")
        found = [ids.get(bytes(t["bytes"])) for t in (c.get("logprobs") or {}).get("content") or []]
        if None in found:
            raise SystemExit(f"{p['id']}: a token whose bytes the vocabulary does not hold")
        rows.append(report.output_row({k: v for k, v in p.items() if k not in ("messages", "body")}, "length" if c["finish_reason"] == "length" else "stop",
                                      found, c["message"].get("content") or ""))
        print(p["id"], rows[-1]["finish"], len(found), flush=True)
    return rows


def gen(url, model, prompts, tokenizer, out, max_tokens=report.MAX_TOKENS):
    """Greedy on the server at url (its model named model): the prompts of a JSON list, the vocabulary of a Hugging Face folder;
    out is JSONL."""
    from transformers import AutoTokenizer
    ids = byte_ids(AutoTokenizer.from_pretrained(tokenizer).get_vocab())
    report.write_lines(out, generate(url, model, report.read(prompts), ids, int(max_tokens)))
