"""
L1-full — Stanza pipeline `tokenize,pos,lemma` с GPU batching.

Что отличается от первой версии:
  • neural lemma processor включён (как требует методичка)
  • используется stanza.bulk_process — обрабатываем сразу пачку Document'ов
    за один прогон по GPU (в 5–10 раз быстрее одиночных nlp(text))
  • длинные тексты режутся по предложениям (re.split по .!?), а не по символам:
    Stanza имеет O(L²) на длине предложения, поэтому короткие чанки решают всё

Запуск:
  CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_gpu_L1full.py --shard 0 --total 3
"""
import argparse
import os
import random
import re
import time
from pathlib import Path

import stanza
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus_raw"
CLEAN = ROOT / "corpus_clean_L1"
CLEAN.mkdir(parents=True, exist_ok=True)

SECTIONS_PRIORITY = [
    ("wiki/articles", None),
    ("theses", None),
    ("nonfiction", None),
    ("umlib_oar", None),
    ("press_mt", 8000),
    ("blogs", 8000),
]

# По методичке, строго
POS_REPLACE = {"PROPN": "PERSON1", "PRON": "PRON1", "NUM": "ORDINAL1"}
POS_DROP = {"PUNCT", "SYM", "X", "SPACE"}

ALPHA_MT_FULL = re.compile(r"^[a-zA-ZċġħżĊĠĦŻ][a-zA-ZċġħżĊĠĦŻ\-']*$")
WIKI_GARBAGE = re.compile(r"[\|\[\]\{\}=*<>#_/\\^~`@$%&]+")
NUMSUFFIX = re.compile(r"\b\d+[a-zA-Z]+\b")
# режем по концу предложения (.!?:) с пробелом/переносом, ИЛИ просто если предложение
# выходит за лимит
SENT_END = re.compile(r"(?<=[.!?:])\s+")

# Целевая длина одного чанка-предложения (Stanza имеет O(L²) на нём)
MAX_SENT_CHARS = 1500

def pre_clean(text: str) -> str:
    text = WIKI_GARBAGE.sub(" ", text)
    text = NUMSUFFIX.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_to_chunks(text: str):
    """Разрезаем текст на чанки <= MAX_SENT_CHARS, стараясь не рвать предложения."""
    text = pre_clean(text)
    if not text:
        return []
    # сначала по предложениям
    sents = SENT_END.split(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        s = s.strip()
        if not s: continue
        # если само предложение очень длинное — режем тупо по словам
        if len(s) > MAX_SENT_CHARS:
            words = s.split()
            piece = []
            piece_len = 0
            for w in words:
                if piece_len + len(w) + 1 > MAX_SENT_CHARS and piece:
                    chunks.append(" ".join(piece))
                    piece, piece_len = [], 0
                piece.append(w); piece_len += len(w) + 1
            if piece:
                chunks.append(" ".join(piece))
            continue
        # обычное предложение — добавляем в текущий чанк
        if cur_len + len(s) + 1 > MAX_SENT_CHARS and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s); cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def normalize_lemma(lemma: str) -> str:
    if not lemma: return ""
    lemma = lemma.lower().strip()
    if not lemma or len(lemma) > 40: return ""
    if not ALPHA_MT_FULL.match(lemma): return ""
    return lemma

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
    ap.add_argument("--batch-docs", type=int, default=16,
                    help="документов в одном bulk_process")
    args = ap.parse_args()

    print(f"=== L1-full shard {args.shard}/{args.total} ===")
    plan = collect_plan(args.shard, args.total)
    if args.smoke:
        plan = plan[:args.smoke]
    print(f"Files in shard: {len(plan)} | batch_docs={args.batch_docs}")

    for sect, _ in SECTIONS_PRIORITY:
        (CLEAN / sect.replace("/", "_")).mkdir(parents=True, exist_ok=True)

    all_lines = CLEAN / f"all_clean.shard{args.shard}.txt"
    index = CLEAN / f"_doc_index.shard{args.shard}.tsv"

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

    t0 = time.time()
    n_done = 0
    n_words = 0

    fall = open(all_lines, "w", encoding="utf-8")
    findex = open(index, "w", encoding="utf-8")
    findex.write("source\tname\tn_words\n")

    pbar = tqdm(total=len(plan), desc=f"L1-shard{args.shard}")
    # Группируем файлы в батчи. Один файл = одна строка в выходном all_clean.
    for i in range(0, len(plan), args.batch_docs):
        batch = plan[i:i+args.batch_docs]
        # Подготовка списка stanza Document'ов: каждый Document = один чанк
        # Для каждого файла собираем чанки и помним границы.
        chunks_per_file = []
        flat_texts = []
        for source, fp in batch:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                chunks_per_file.append((source, fp, 0)); continue
            chs = split_to_chunks(text)
            chunks_per_file.append((source, fp, len(chs)))
            flat_texts.extend(chs)

        if not flat_texts:
            pbar.update(len(batch))
            continue

        # bulk_process через создание Document'ов
        try:
            docs_in = [stanza.Document([], text=t) for t in flat_texts]
            docs_out = nlp(docs_in)
        except Exception as e:
            print(f"\n[!] batch failed: {e}")
            pbar.update(len(batch))
            continue

        # Раскладываем результаты обратно по файлам
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
                        tok = normalize_lemma(w.lemma or w.text)
                        if tok: words.append(tok)
            if not words:
                pbar.update(1); continue
            clean = " ".join(words)
            (CLEAN / source / f"{fp.stem}.txt").write_text(clean, encoding="utf-8")
            fall.write(clean + "\n")
            findex.write(f"{source}\t{fp.stem}\t{len(words)}\n")
            n_done += 1; n_words += len(words)
            pbar.update(1)

    pbar.close()
    fall.close(); findex.close()
    el = time.time() - t0
    print(f"\nL1-full shard {args.shard}: done={n_done} words={n_words} elapsed={el/60:.1f}min "
          f"speed={(n_done/el if el>0 else 0):.2f} docs/s")

if __name__ == "__main__":
    main()
