"""
Скачивает выбранные секции Korpus Malti v4.2 с HuggingFace
(MLRS/korpus_malti) и сохраняет каждый документ как отдельный .txt
в corpus_raw/<section>/.

Приоритет — литература и связные тексты, как требует методичка:
  belles_lettres   — художественная литература (главное!)
  nonfiction       — нон-фикшн
  theses           — академические работы
  umlib_oar        — open-access публикации UM
  press_mt         — мальтийская пресса
  blogs            — блоги

Skip: parliament, government_*, law_*, court — формальные тексты,
неподходящие для оценки «естественности» языка.
"""
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUT_ROOT = Path(__file__).resolve().parent.parent / "corpus_raw"
SECTIONS = [
    "nonfiction",   # нон-фикшн (длинные связные тексты)
    "theses",       # академические диссертации
    "umlib_oar",    # open-access publications UM
    "press_mt",     # мальтийская пресса
    "blogs",        # блоги
]
# belles_lettres — пустая на HF (copyright, нужен отдельный запрос в MLRS)
MIN_CHARS = 500
SAFE = re.compile(r"[^\w\-\.() ]+", flags=re.UNICODE)

def safe_name(s: str) -> str:
    n = SAFE.sub("_", s).strip().strip(".")
    return n[:120] or "doc"

def write_section(section: str):
    out = OUT_ROOT / section
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {section} ===")
    ds = load_dataset("MLRS/korpus_malti", section, split="train")
    print(f"  examples: {len(ds)}")
    seen = set()
    written = 0
    skipped = 0
    for i, row in enumerate(tqdm(ds, desc=section)):
        text_field = row.get("text")
        # text может быть list[str] (sentences) или str
        if isinstance(text_field, list):
            body = " ".join(t for t in text_field if t)
        else:
            body = text_field or ""
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) < MIN_CHARS:
            skipped += 1; continue
        # имя: пробуем metadata, иначе индекс
        meta_keys = [k for k in row.keys() if k != "text"]
        title = None
        for k in ("title", "name", "doc_id", "id", "url"):
            if k in row and row[k]:
                title = str(row[k]); break
        base = safe_name(title) if title else f"{section}_{i:06d}"
        cand = base; n = 2
        while cand in seen:
            cand = f"{base}_{n}"; n += 1
        seen.add(cand)
        (out / f"{cand}.txt").write_text(body, encoding="utf-8")
        written += 1
    print(f"  wrote: {written}, skipped: {skipped}")
    return written, skipped

def main():
    totals = {}
    for sec in SECTIONS:
        try:
            totals[sec] = write_section(sec)
        except Exception as e:
            print(f"  ERROR for {sec}: {e}")
            totals[sec] = (0, 0)
    print("\n=== SUMMARY ===")
    for sec, (w, s) in totals.items():
        print(f"  {sec:18s}  written={w:>8d}  skipped={s:>8d}")

if __name__ == "__main__":
    main()
