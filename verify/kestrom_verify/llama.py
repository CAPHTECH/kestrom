"""llama.cpp: decoding with llama-server (Homebrew, 0.9.0) on a GGUF file, from the prompt token ids. The range of a GGUF row edit
is weights.diff; the comparison is report.greedy."""
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import report

PORT = 9931
GREEDY = (1, dict(n_predict=report.MAX_TOKENS, temperature=0, top_k=1))  # one slot


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def generate(model, prompts, decode, slots, request, log):
    """Per prompt ({"ids": prompt ids, other keys kept}): finish (stop or length), the tokens without the stop token, their text
    (decode). The server's messages go to log."""
    server = subprocess.Popen(["llama-server", "-m", str(model), "--port", str(PORT), "--host", "127.0.0.1", "-np", str(slots), "-c", str(8192 * slots), "-ngl", "999", "--seed", "0"],
                              stdout=subprocess.DEVNULL, stderr=open(log, "w"))
    try:
        for _ in range(600):
            try:
                if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).read()).get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError):  # not up yet
                time.sleep(2)

        def one(p):
            a = post("/completion", dict(prompt=p["ids"], cache_prompt=False, return_tokens=True, **request))
            toks = a["tokens"][:-1] if a["stop_type"] == "eos" else a["tokens"]  # the stop token is the last one
            print(p["id"], flush=True)
            return dict({k: v for k, v in p.items() if k != "ids"}, finish="length" if a["stop_type"] == "limit" else "stop", tokens=toks, text=decode(toks))
        with ThreadPoolExecutor(slots) as pool:
            return list(pool.map(one, prompts))
    finally:
        server.terminate()
        server.wait()


def gen(model, prompts, tokenizer, out):
    """Greedy: the prompts of a JSON list, decoded with the tokenizer of a Hugging Face folder; out is JSONL."""
    from transformers import AutoTokenizer
    report.write_lines(out, generate(model, report.read(prompts), AutoTokenizer.from_pretrained(tokenizer).decode, *GREEDY, f"{out}.server.log"))
