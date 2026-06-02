"""
GPU-препроцессинг V2 — со «своим» лемматизатором на базе Ġabra (UM MLRS).

Цепочка лемматизации каждого слова:
  1) точный lookup в Ġabra (mt_lemma_lookup.json, ~1.3M словоформ)
  2) lookup без диакритик (та же таблица уже содержит варианты)
  3) lookup без артикля/префикса (правила-стрипперы)
  4) fallback: лемма из Stanza
  5) если и она пустая — surface form

POS-замены строго по методичке Spot the bot:
  PROPN → PERSON1
  PRON  → PRON1
  NUM   → ORDINAL1
  PUNCT/SYM/X — выбрасываем

Запуск:
  CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_gpu_v2.py --shard 0 --total 3
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

# Строго по методичке
POS_REPLACE = {"PROPN": "PERSON1", "PRON": "PRON1", "NUM": "ORDINAL1"}
POS_DROP = {"PUNCT", "SYM", "X", "SPACE"}

ALPHA_MT_FULL = re.compile(r"^[a-zA-ZċġħżĊĠĦŻ][a-zA-ZċġħżĊĠĦŻ\-']*$")
WIKI_GARBAGE = re.compile(r"[\|\[\]\{\}=*<>#_/\\^~`@$%&]+")
NUMSUFFIX = re.compile(r"\b\d+[a-zA-Z]+\b")

# Мальтийские артикли и проклитики (срастаются с существительным через дефис/апостроф)
ARTICLE_PREFIXES = [
    "il-", "it-", "id-", "in-", "ir-", "is-", "iz-", "iż-", "ix-",
    "l-", "t-", "d-", "n-", "r-", "s-", "z-", "ż-", "x-",
]
PROCLITIC_APOS = ["b'", "f'", "m'", "t'", "s'", "n'", "x'", "ġ'"]


def strip_clitics(word: str):
    """Снять артикли и проклитики. Возвращает (stripped, was_modified)."""
    w = word
    changed = False

    # сначала проклитики с апострофом
    for p in PROCLITIC_APOS:
        if w.lower().startswith(p):
            w = w[len(p):]
            changed = True; break

    # потом артикли с дефисом
    for p in ARTICLE_PREFIXES:
        if w.lower().startswith(p):
            w = w[len(p):]
            changed = True; break
    return w, changed


DIA = str.maketrans({"ċ":"c", "ġ":"g", "ħ":"h", "ż":"z",
                     "Ċ":"C", "Ġ":"G", "Ħ":"H", "Ż":"Z"})

# Загрузим Ġabra-lookup
_LOOKUP = None


def get_lookup():
    global _LOOKUP
    if _LOOKUP is None:
        p = RES / "mt_lemma_lookup.json"
        with open(p, "r", encoding="utf-8") as f:
            _LOOKUP = json.load(f)
        print(f"[gabra] loaded {len(_LOOKUP)} wordforms")
    return _LOOKUP


def custom_lemma(surface: str, stanza_lemma: str | None = None) -> str:
    """Наш собственный лемматизатор: Ġabra + правила, fallback на Stanza."""
    lk = get_lookup()
    s = surface.lower().strip()
    if not s:
        return ""

    # 1) прямой lookup
    if s in lk:
        return lk[s]

    # 2) lookup без диакритик (lookup уже содержит такие ключи, но на всякий)
    s_nd = s.translate(DIA)
    if s_nd in lk:
        return lk[s_nd]

    # 3) снимаем артикль/проклитику и пробуем ещё раз
    stripped, changed = strip_clitics(s)
    if changed and stripped in lk:
        return lk[stripped]
    if changed and stripped.translate(DIA) in lk:
        return lk[stripped.translate(DIA)]

    # 4) Stanza-лемма
    if stanza_lemma:
        sl = stanza_lemma.lower().strip()
        if sl:
            return sl

    # 5) сама форма
    return s


def normalize_lemma(lemma: str) -> str:
    if not lemma:
        return ""
    lemma = lemma.lower().strip()
    if not lemma or len(lemma) > 40:
        return ""
    if not ALPHA_MT_FULL.match(lemma):
        return ""
    return lemma


def pre_clean(text: str) -> str:
    text = WIKI_GARBAGE.sub(" ", text)
    text = NUMSUFFIX.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 10000):
    if len(text) <= max_chars:
        return [text]
    chunks, buf, cur = [], [], 0
    for para in re.split(r"\n\s*\n", text):
        if cur + len(para) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf)); buf, cur = [], 0
        buf.append(para); cur += len(para) + 2
    if buf: chunks.append("\n\n".join(buf))
    return chunks


def process_doc(nlp, text: str, stats: dict):
    text = pre_clean(text)
    out_words = []
    for ch in chunk_text(text):
        try:
            doc = nlp(ch)
        except Exception:
            continue
        for sent in doc.sentences:
            for w in sent.words:
                up = w.upos
                if up in POS_DROP: continue
                if up in POS_REPLACE:
                    out_words.append(POS_REPLACE[up]); continue

                # custom lemmatizer
                lem = custom_lemma(w.text, w.lemma)

                # отчёт — какой ярус сработал
                if lem == w.text.lower():
                    stats["from_surface"] += 1
                elif (w.lemma or "").lower() == lem:
                    stats["from_stanza"] += 1
                else:
                    stats["from_gabra"] += 1
                tok = normalize_lemma(lem)
                if tok: out_words.append(tok)
    return out_words


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
    args = ap.parse_args()

    print(f"=== V2 shard {args.shard}/{args.total} ===")
    plan = collect_plan(args.shard, args.total)
    if args.smoke:
        plan = plan[:args.smoke]
    print(f"Files in shard: {len(plan)}")

    for sect, _ in SECTIONS_PRIORITY:
        (CLEAN / sect.replace("/", "_")).mkdir(parents=True, exist_ok=True)

    all_lines = CLEAN / f"all_clean.shard{args.shard}.txt"
    index = CLEAN / f"_doc_index.shard{args.shard}.tsv"

    get_lookup()  # прелоад

    nlp = stanza.Pipeline(
        lang="mt",
        processors="tokenize,pos,lemma",
        use_gpu=True,
        verbose=False,
        download_method=None,
    )

    stats = {"from_gabra": 0, "from_stanza": 0, "from_surface": 0}
    t0 = time.time()
    n_done = 0
    n_words = 0

    with open(all_lines, "w", encoding="utf-8") as fall, \
         open(index, "w", encoding="utf-8") as findex:
        findex.write("source\tname\tn_words\n")
        for source, fp in tqdm(plan, desc=f"shard{args.shard}"):
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text: continue
            words = process_doc(nlp, text, stats)
            if not words: continue
            clean = " ".join(words)
            (CLEAN / source / f"{fp.stem}.txt").write_text(clean, encoding="utf-8")
            fall.write(clean + "\n")
            findex.write(f"{source}\t{fp.stem}\t{len(words)}\n")
            n_done += 1; n_words += len(words)

    el = time.time() - t0
    print(f"\nshard {args.shard}: done={n_done} words={n_words} elapsed={el/60:.1f}min")
    total = sum(stats.values()) or 1
    print(f"lemma sources: gabra={stats['from_gabra']} ({100*stats['from_gabra']/total:.1f}%) "
          f"stanza={stats['from_stanza']} ({100*stats['from_stanza']/total:.1f}%) "
          f"surface={stats['from_surface']} ({100*stats['from_surface']/total:.1f}%)")


if __name__ == "__main__":
    main()
