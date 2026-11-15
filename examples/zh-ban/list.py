"""The ban list of zh-ban for any tokenizer: one "id:N" line per token, for kestrom rows ban.
Usage: python examples/zh-ban/list.py TOKENIZER JA.parquet ZH.parquet OUT.txt
  TOKENIZER: a Hugging Face folder or name. JA, ZH: Wikipedia shards with a text column; list.txt was made from wikimedia/wikipedia
  20231101.ja/train-00000-of-00015.parquet and 20231101.zh/train-00002-of-00006.parquet with Qwen3.5-9B's tokenizer.
A token is listed when (1) it holds a han character outside JIS X 0213, or (2) it holds han and no kana, is seen at least 20 times
in the Chinese articles and at most 1/100 as often in the Japanese ones, and does not read as Japanese (EXCLUDE, read on Qwen3.5-9B
outputs). The counts run over 20,000 random articles of each shard cut to 2,000 characters, plus the first 5,000 characters of
every medical article (4 of the medical words). Tokens that decode to a piece of a character (byte pieces) are not listed: a model
can still write a banned character byte by byte. Read the list before you ship a model."""
import random
import sys
from collections import Counter

ZH_MIN, ZH_RATIO, ARTICLES, CHARS, MED_MIN, MED_CHARS = 20, 100, 20000, 2000, 4, 5000
MED = {"ja": ("症状", "治療", "患者", "診断", "疾患", "投与", "感染", "臨床"),
       "zh": ("症状", "治疗", "患者", "诊断", "疾病", "给药", "感染", "临床", "症狀", "治療", "診斷", "給藥", "臨床")}
EXCLUDE = {"座位", "代表性", "弯", "部的", "河谷", "菲", "科的", "数量", "或", "渣", "裡", "的小", "的最大", "的心", "埃"}


def han(c):
    o = ord(c)
    return 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x3134F


def jis(c):
    try:
        c.encode("euc_jis_2004")
        return True
    except UnicodeEncodeError:
        return False


def counts(tok, path, lang):
    import pyarrow.parquet as pq
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    sample = [texts[i][:CHARS] for i in random.Random(0).sample(range(len(texts)), ARTICLES)]
    med = [t[:MED_CHARS] for t in texts if sum(w in t for w in MED[lang]) >= MED_MIN]
    cnt = Counter()
    for part in (sample, med):
        for i in range(0, len(part), 500):
            for ids in tok(part[i:i + 500], add_special_tokens=False).input_ids:
                cnt.update(ids)
    return cnt


def main(tokenizer, ja_file, zh_file, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    ja, zh = counts(tok, ja_file, "ja"), counts(tok, zh_file, "zh")
    nja, nzh = sum(ja.values()), sum(zh.values())
    nonjis, words = [], []
    for i in range(len(tok)):
        s = tok.decode([i])
        if "�" in s:
            continue
        if any(han(c) and not jis(c) for c in s):
            nonjis.append(i)
        elif (any(han(c) for c in s) and not any(0x3040 <= ord(c) <= 0x30FF for c in s) and zh[i] >= ZH_MIN
              and ja[i] / nja <= zh[i] / nzh / ZH_RATIO and s not in EXCLUDE):
            words.append(i)
    open(out, "w", encoding="utf-8").write("".join(f"id:{i}\n" for i in sorted(nonjis + words)))
    print(len(nonjis), "tokens outside JIS X 0213,", len(words), "Chinese words")


if __name__ == "__main__":
    main(*sys.argv[1:])
