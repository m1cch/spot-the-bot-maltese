"""
L2 — Stanza pipeline + наш Ġabra-lookup лемматизатор.

Идентичен L1-full по производительности (bulk_process + split by sentence),
но после Stanza-лемматизации применяется наш custom_lemma:
  1) точный lookup в Ġabra (mt_lemma_lookup.json, ~1.3M словоформ)
  2) lookup без диакритик (уже в той же таблице)
  3) lookup после снятия артикля/проклитики
  4) fallback на Stanza-лемму
  5) если ничего — сама surface form
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import stanza
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus_raw"
CLEAN = ROOT / "corpus_clean_L2"
RES = ROOT / "resources"
CLEAN.mkdir(parents=True, exist_ok=True)

SECTIONS_PRIORITY = [
    ("wiki/articles", None),
    ("theses", None),
    ("nonfiction", None),
    ("umlib_oar", None),
    ("press_mt", 8000),
    ("blogs", 8000),
]

POS_REPLACE = {"PROPN": "PERSON1", "PRON": "PRON1", "NUM": "ORDINAL1"}
POS_DROP = {"PUNCT", "SYM", "X", "SPACE"}

ALPHA_MT_FULL = re.compile(r"^[a-zA-ZċġħżĊĠĦŻ][a-zA-ZċġħżĊĠĦŻ\-']*$")
WIKI_GARBAGE = re.compile(r"[\|\[\]\{\}=*<>#_/\\^~`@$%&]+")
NUMSUFFIX = re.compile(r"\b\d+[a-zA-Z]+\b")
SENT_END = re.compile(r"(?<=[.!?:])\s+")
MAX_SENT_CHARS = 1500

# Mаltese morphology helpers
ARTICLE_PREFIXES = ("il-", "it-", "id-", "in-", "ir-", "is-", "iz-", "iż-", "ix-",
                    "iċ-", "iġ-", "iħ-",
                    "l-", "t-", "d-", "n-", "r-", "s-", "z-", "ż-", "x-",
                    "ċ-", "ġ-", "ħ-")
PROCLITIC_APOS = ("b'", "f'", "m'", "t'", "s'", "n'", "x'", "ġ'", "ż'", "ċ'", "ħ'",
                  "k'", "p'")
DIA = str.maketrans({"ċ":"c", "ġ":"g", "ħ":"h", "ż":"z",
                     "Ċ":"C", "Ġ":"G", "Ħ":"H", "Ż":"Z"})


def strip_clitics(word: str):
    wl = word.lower()
    for p in PROCLITIC_APOS:
        if wl.startswith(p):
            return word[len(p):], True
    for p in ARTICLE_PREFIXES:
        if wl.startswith(p):
            return word[len(p):], True
    return word, False


_LOOKUP = None


def get_lookup():
    global _LOOKUP
    if _LOOKUP is None:
        with open(RES / "mt_lemma_lookup.json", "r", encoding="utf-8") as f:
            _LOOKUP = json.load(f)
        print(f"[gabra] loaded {len(_LOOKUP)} wordforms")
    return _LOOKUP


def custom_lemma(surface: str, stanza_lemma: str | None) -> str:
    lk = get_lookup()
    s = (surface or "").lower().strip()
    if not s:
        return ""
    if s in lk: return lk[s]
    s_nd = s.translate(DIA)
    if s_nd in lk: return lk[s_nd]
    stripped, changed = strip_clitics(s)
    if changed and stripped in lk: return lk[stripped]
    if changed and stripped.translate(DIA) in lk: return lk[stripped.translate(DIA)]
    if stanza_lemma:
        sl = stanza_lemma.lower().strip()
        if sl: return sl
    return s


def normalize_lemma(lemma: str) -> str:
    if not lemma: return ""
    lemma = lemma.lower().strip()
    if not lemma or len(lemma) > 40: return ""
    if not ALPHA_MT_FULL.match(lemma): return ""
    return lemma


def pre_clean(text: str) -> str:
    text = WIKI_GARBAGE.sub(" ", text)
    text = NUMSUFFIX.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_to_chunks(text: str):
    text = pre_clean(text)
    if not text: return []
    sents = SENT_END.split(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        s = s.strip()
        if not s: continue
        if len(s) > MAX_SENT_CHARS:
            words = s.split()
            piece, piece_len = [], 0
            for w in words:
                if piece_len + len(w) + 1 > MAX_SENT_CHARS and piece:
                    chunks.append(" ".join(piece)); piece, piece_len = [], 0
                piece.append(w); piece_len += len(w) + 1
            if piece: chunks.append(" ".join(piece))
            continue
        if cur_len + len(s) + 1 > MAX_SENT_CHARS and cur:
            chunks.append(" ".join(cur)); cur, cur_len = [], 0
        cur.append(s); cur_len += len(s) + 1
    if cur: chunks.append(" ".join(cur))
    return chunks


def collect_plan(shard: int, total: int):
    plan = []
    for sect, cap in SECTIONS_PRIORITY:
        src = RAW / sect
        if not src.exists():
            print(f"  skip {sect}: not found"); continue
        files = sorted(src.glob("*.txt"))
        if cap is not None and len(files) > cap:
            rng = random.Random(42)
            files = rng.sample(files, cap)
        section_name = sect.replace("/", "_")
        for fp in files:
            plan.append((section_name, fp))
    plan = [p for i, p in enumerate(plan) if i % total == shard]
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--batch-docs", type=int, default=16)
    args = ap.parse_args()

    print(f"=== L2 (Stanza+Ġabra) shard {args.shard}/{args.total} ===")
    plan = collect_plan(args.shard, args.total)
    if args.smoke:
        plan = plan[:args.smoke]
    print(f"Files in shard: {len(plan)} | batch_docs={args.batch_docs}")

    for sect, _ in SECTIONS_PRIORITY:
        (CLEAN / sect.replace("/", "_")).mkdir(parents=True, exist_ok=True)

    all_lines = CLEAN / f"all_clean.shard{args.shard}.txt"
    index = CLEAN / f"_doc_index.shard{args.shard}.tsv"

    get_lookup()  # preload Gabra

    nlp = stanza.Pipeline(
        lang="mt",
        processors="tokenize,pos,lemma",
        use_gpu=True,
        verbose=False,
        download_method=None,
        tokenize_batch_size=128,
        pos_batch_size=128,
        lemma_batch_size=64,
    )

    stats = {"from_gabra": 0, "from_stanza": 0, "from_surface": 0}
    t0 = time.time()
    n_done = 0
    n_words = 0

    fall = open(all_lines, "w", encoding="utf-8")
    findex = open(index, "w", encoding="utf-8")
    findex.write("source\tname\tn_words\n")

    pbar = tqdm(total=len(plan), desc=f"L2-shard{args.shard}")
    for i in range(0, len(plan), args.batch_docs):
        batch = plan[i:i+args.batch_docs]
        chunks_per_file, flat_texts = [], []
        for source, fp in batch:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                chunks_per_file.append((source, fp, 0)); continue
            chs = split_to_chunks(text)
            chunks_per_file.append((source, fp, len(chs)))
            flat_texts.extend(chs)
        if not flat_texts:
            pbar.update(len(batch)); continue
        try:
            docs_in = [stanza.Document([], text=t) for t in flat_texts]
            docs_out = nlp(docs_in)
        except Exception as e:
            print(f"\n[!] batch failed: {e}")
            pbar.update(len(batch)); continue

        cursor = 0
        for source, fp, n_chunks in chunks_per_file:
            if n_chunks == 0:
                pbar.update(1); continue
            file_docs = docs_out[cursor:cursor+n_chunks]
            cursor += n_chunks
            words = []
            for d in file_docs:
                for sent in d.sentences:
                    for w in sent.words:
                        up = w.upos
                        if up in POS_DROP: continue
                        if up in POS_REPLACE:
                            words.append(POS_REPLACE[up]); continue
                        lem = custom_lemma(w.text, w.lemma)
                        if lem == w.text.lower():
                            stats["from_surface"] += 1
                        elif (w.lemma or "").lower() == lem:
                            stats["from_stanza"] += 1
                        else:
                            stats["from_gabra"] += 1
                        tok = normalize_lemma(lem)
                        if tok: words.append(tok)
            if not words:
                pbar.update(1); continue
            clean = " ".join(words)
            (CLEAN / source / f"{fp.stem}.txt").write_text(clean, encoding="utf-8")
            fall.write(clean + "\n")
            findex.write(f"{source}\t{fp.stem}\t{len(words)}\n")
            n_done += 1; n_words += len(words)
            pbar.update(1)

    pbar.close(); fall.close(); findex.close()
    el = time.time() - t0
    total = sum(stats.values()) or 1
    print(f"\nL2 shard {args.shard}: done={n_done} words={n_words} elapsed={el/60:.1f}min")
    print(f"  lemma sources: gabra={100*stats['from_gabra']/total:.1f}%  "
          f"stanza={100*stats['from_stanza']/total:.1f}%  "
          f"surface={100*stats['from_surface']/total:.1f}%")


if __name__ == "__main__":
    main()
