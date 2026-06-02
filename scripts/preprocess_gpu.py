"""
GPU-препроцессинг мальтийских текстов.

Stanza pipeline (tokenize → pos → lemma) на одной CUDA-карте.
Поддерживает sharding на N процессов через --shard и --total.

Запуск (одна карта, всё):
  CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_gpu.py

Параллельно на 3 GPU:
  CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_gpu.py --shard 0 --total 3 &
  CUDA_VISIBLE_DEVICES=1 python scripts/preprocess_gpu.py --shard 1 --total 3 &
  CUDA_VISIBLE_DEVICES=2 python scripts/preprocess_gpu.py --shard 2 --total 3 &
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

# (имя секции, cap)
SECTIONS_PRIORITY = [
    ("wiki/articles", None),
    ("theses", None),
    ("nonfiction", None),
    ("umlib_oar", None),
    ("press_mt", 8000),    # cap, иначе очень долго
    ("blogs", 8000),
]

POS_REPLACE = {
    "PROPN": "PERSON1",
    "PRON":  "PRON1",
    "NUM":   "ORDINAL1",
}
POS_DROP = {"PUNCT", "SYM", "X", "SPACE"}
# Полностью допустимый токен: только латиница (с мальт. диакритиками), дефис, апостроф
ALPHA_MT_FULL = re.compile(r"^[a-zA-ZċġħżĊĠĦŻ][a-zA-ZċġħżĊĠĦŻ\-']*$")
# Wiki markup и мусор для pre-clean
WIKI_GARBAGE = re.compile(r"[\|\[\]\{\}=*<>#_/\\^~`@$%&]+")
# px-suffix, hex-colors, etc
NUMSUFFIX = re.compile(r"\b\d+[a-zA-Z]+\b")


def pre_clean(text: str) -> str:
    """Чистим wiki-markup рудименты и явный мусор ДО stanza."""
    text = WIKI_GARBAGE.sub(" ", text)
    text = NUMSUFFIX.sub(" ", text)

    # сжать пробелы, оставить переводы строк
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_lemma(lemma: str) -> str:
    """Разрешаем только буквы (с мальт. диакритиками), дефис, апостроф. Иначе токен — мусор."""
    if not lemma:
        return ""
    lemma = lemma.lower().strip()
    if not lemma or len(lemma) > 40:
        return ""
    if not ALPHA_MT_FULL.match(lemma):
        return ""
    return lemma


def collect_plan(shard: int = 0, total: int = 1):
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

    # шардинг: каждый процесс берёт каждую total-ю запись со смещением shard
    plan = [p for i, p in enumerate(plan) if i % total == shard]
    return plan


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


def process_doc(nlp, text: str):
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
                tok = normalize_lemma(w.lemma or w.text)
                if tok: out_words.append(tok)
    return out_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--smoke", type=int, default=0, help="N files for quick speed test")
    args = ap.parse_args()

    print(f"=== shard {args.shard}/{args.total} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')} ===")
    plan = collect_plan(args.shard, args.total)
    if args.smoke:
        plan = plan[:args.smoke]
    print(f"Files in this shard: {len(plan)}")

    # подготовим директории
    for sect, _ in SECTIONS_PRIORITY:
        (CLEAN / sect.replace("/", "_")).mkdir(parents=True, exist_ok=True)

    # одно общее имя файлов для шарда — у каждого свой суффикс
    all_lines_path = CLEAN / f"all_clean.shard{args.shard}.txt"
    index_path = CLEAN / f"_doc_index.shard{args.shard}.tsv"

    import torch
    print(f"torch.cuda.is_available={torch.cuda.is_available()} dev={torch.cuda.current_device() if torch.cuda.is_available() else None}")

    # L1 = «минимальная Stanza»: tokenize + POS (без neural lemma — она ОЧЕНЬ медленная).
    # Лемма = surface form в нижнем регистре. Это будет baseline для сравнения с L2 (Ġabra)
    # и L3 (rule-based).
    nlp = stanza.Pipeline(
        lang="mt",
        processors="tokenize,pos",
        use_gpu=True,
        verbose=False,
        download_method=None,
        tokenize_batch_size=128,
        pos_batch_size=128,
    )

    t0 = time.time()
    n_words_total = 0
    n_done = 0

    with open(all_lines_path, "w", encoding="utf-8") as fall, \
         open(index_path, "w", encoding="utf-8") as findex:
        findex.write("source\tname\tn_words\n")
        for source, fp in tqdm(plan, desc=f"shard{args.shard}"):
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text: continue
            words = process_doc(nlp, text)
            if not words: continue
            clean = " ".join(words)
            (CLEAN / source / f"{fp.stem}.txt").write_text(clean, encoding="utf-8")
            fall.write(clean + "\n")
            findex.write(f"{source}\t{fp.stem}\t{len(words)}\n")
            n_done += 1; n_words_total += len(words)

    el = time.time() - t0
    print(f"\nshard {args.shard}: done={n_done} words={n_words_total} elapsed={el/60:.1f}min "
          f"speed={(n_done/el) if el>0 else 0:.2f} docs/s")


if __name__ == "__main__":
    main()
