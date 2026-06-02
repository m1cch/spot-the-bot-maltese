"""
Препроцессинг мальтийских текстов по методичке Spot the bot.

Применяет Stanza-пайплайн (tokenize → pos → lemma):
  - PROPN  → PERSON1
  - PRON   → PRON1
  - NUM    → ORDINAL1
  - DET    → DET1  (артикли в мальтийском — часто связаны с существительным)
  - PUNCT  → выбрасывается
  - SYM, X → выбрасывается
  - остальные слова заменяются леммой (нижний регистр)

Стратегия сэмплирования (для 12-дневного дедлайна на CPU):
  - wiki         — целиком (~7k файлов)
  - umlib_oar    — целиком (~11k, длинные тексты)
  - nonfiction   — целиком (~2k)
  - theses       — целиком (18)
  - press_mt     — первые 5k файлов (журналистика — много, но похожа)
  - blogs        — первые 5k файлов

На выходе:
  corpus_clean/<source>/<orig_name>.txt   — один текст на файл, слова через пробел
  corpus_clean/all_clean.txt              — единый файл, строка = один текст (для SVD)
  corpus_clean/_doc_index.tsv             — manifest (source, name, n_words)

Запуск:
  python scripts/preprocess.py [--limit-per-section N] [--workers N]
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import stanza
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus_raw"
CLEAN = ROOT / "corpus_clean"
CLEAN.mkdir(parents=True, exist_ok=True)

SECTIONS_PRIORITY = [
    ("wiki/articles", None),     # all
    ("theses", None),            # all (only 18)
    ("nonfiction", None),        # all
    ("umlib_oar", None),         # all
    ("press_mt", 5000),          # cap
    ("blogs", 5000),             # cap
]

# Замены POS
POS_REPLACE = {
    "PROPN": "PERSON1",
    "PRON":  "PRON1",
    "NUM":   "ORDINAL1",
    "DET":   "DET1",
}
POS_DROP = {"PUNCT", "SYM", "X", "SPACE"}

# Регулярка для отсеивания мусора в лемме
ALPHANUM_MT = re.compile(r"[a-zA-Z0-9ċġħżĊĠĦŻ\-']+")


def normalize_lemma(lemma: str) -> str:
    if not lemma:
        return ""
    lemma = lemma.lower().strip()

    # отбрасываем явный мусор
    if not lemma or len(lemma) > 40:
        return ""

    # должно содержать хоть одну букву/цифру
    if not ALPHANUM_MT.search(lemma):
        return ""
    return lemma


# Глобальный nlp в воркере
_NLP = None


def _init_worker():
    global _NLP
    _NLP = stanza.Pipeline(
        lang="mt",
        processors="tokenize,pos,lemma",
        use_gpu=False,
        verbose=False,
        download_method=None,
    )


def _process_one(args):
    """Обрабатывает один файл. Возвращает (source, name, clean_text, n_words) или None."""
    global _NLP
    source, filepath = args
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None

    # Stanza не любит сверхдлинные тексты — режем по 30k символов на пайплайн-инпут
    chunks = []
    MAX = 30000
    if len(text) <= MAX:
        chunks = [text]
    else:
        # режем по абзацам
        buf = []
        cur = 0
        for para in re.split(r"\n\s*\n", text):
            if cur + len(para) + 2 > MAX and buf:
                chunks.append("\n\n".join(buf))
                buf, cur = [], 0
            buf.append(para); cur += len(para) + 2
        if buf: chunks.append("\n\n".join(buf))
    out_words = []
    try:
        for ch in chunks:
            doc = _NLP(ch)
            for sent in doc.sentences:
                for w in sent.words:
                    upos = w.upos
                    if upos in POS_DROP:
                        continue
                    if upos in POS_REPLACE:
                        out_words.append(POS_REPLACE[upos])
                        continue
                    token = normalize_lemma(w.lemma or w.text)
                    if token:
                        out_words.append(token)
    except Exception:
        return None
    if not out_words:
        return None
    return (source, Path(filepath).stem, " ".join(out_words), len(out_words))


def collect_files():
    plan = []
    for sect, cap in SECTIONS_PRIORITY:
        src_dir = RAW / sect
        if not src_dir.exists():
            print(f"  skip {sect}: not found")
            continue
        files = sorted(src_dir.glob("*.txt"))
        if cap is not None and len(files) > cap:
            # детерминированно: random.seed
            rng = random.Random(42)
            files = rng.sample(files, cap)
        section_name = sect.replace("/", "_")
        for fp in files:
            plan.append((section_name, str(fp)))
        print(f"  {sect:25s} -> {len(files)} files")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--smoke", action="store_true", help="quick test: 30 files")
    args = ap.parse_args()

    print(f"=== Planning ===")
    plan = collect_files()
    if args.smoke:
        plan = plan[:30]
    print(f"Total files to process: {len(plan)}")
    print(f"Workers: {args.workers}")

    # Подготовим выходные папки
    for sect, _ in SECTIONS_PRIORITY:
        (CLEAN / sect.replace("/", "_")).mkdir(parents=True, exist_ok=True)

    all_lines_path = CLEAN / "all_clean.txt"
    index_path = CLEAN / "_doc_index.tsv"

    n_done = 0
    n_words_total = 0
    t0 = time.time()

    with open(all_lines_path, "w", encoding="utf-8") as fall, \
         open(index_path, "w", encoding="utf-8") as findex, \
         ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        findex.write("source\tname\tn_words\n")
        futures = [pool.submit(_process_one, item) for item in plan]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="docs"):
            res = fut.result()
            if res is None:
                continue
            source, name, clean, nw = res

            # сохраняем поштучно
            out_p = CLEAN / source / f"{name}.txt"
            out_p.write_text(clean, encoding="utf-8")

            # единый файл (строка = один текст)
            fall.write(clean.replace("\n", " ") + "\n")

            # индекс
            findex.write(f"{source}\t{name}\t{nw}\n")
            n_done += 1
            n_words_total += nw

    elapsed = time.time() - t0
    print(f"\nDone. Files: {n_done}, words(cleaned): {n_words_total}, elapsed: {elapsed/60:.1f} min")
    print(f"Output: {CLEAN}")


if __name__ == "__main__":
    main()
