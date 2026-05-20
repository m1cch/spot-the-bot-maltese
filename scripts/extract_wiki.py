"""
Извлекает статьи из мальтийского Wikipedia-дампа в отдельные .txt файлы.
Один файл = одна статья, имя файла = заголовок статьи.

Использует потоковый парсинг bz2 + ElementTree, чтобы не грузить всё в память.
mwparserfromhell снимает wiki-разметку.
"""
import bz2
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell
from tqdm import tqdm

DUMP = Path(__file__).resolve().parent.parent / "corpus_raw" / "wiki" / "mtwiki-latest-pages-articles.xml.bz2"
OUT = Path(__file__).resolve().parent.parent / "corpus_raw" / "wiki" / "articles"
OUT.mkdir(parents=True, exist_ok=True)

# Минимальная длина текста после очистки (символов)
MIN_LEN = 500

# Безопасное имя файла
SAFE = re.compile(r"[^\w\-\.() ]+", flags=re.UNICODE)
def safe_name(title: str) -> str:
    name = SAFE.sub("_", title).strip().strip(".")
    return name[:150] or "untitled"

def iter_pages(path: Path):
    """Yield (title, text) для каждой статьи main namespace."""
    with bz2.open(path, "rb") as f:
        ns_map = {}
        for ev, el in ET.iterparse(f, events=("start", "end", "start-ns")):
            if ev == "start-ns":
                prefix, uri = el
                if prefix == "":
                    ns_map["x"] = uri
                continue
            if ev != "end":
                continue
            tag = el.tag.split("}", 1)[-1]
            if tag != "page":
                continue
            ns_el = el.find(f"{{{ns_map.get('x','')}}}ns") if ns_map else el.find("ns")
            title_el = el.find(f"{{{ns_map.get('x','')}}}title") if ns_map else el.find("title")
            rev = el.find(f"{{{ns_map.get('x','')}}}revision") if ns_map else el.find("revision")
            if rev is None:
                el.clear(); continue
            txt = rev.find(f"{{{ns_map.get('x','')}}}text") if ns_map else rev.find("text")
            ns_val = ns_el.text if ns_el is not None else None
            if ns_val != "0":
                el.clear(); continue
            title = title_el.text if title_el is not None else None
            text = txt.text if txt is not None else None
            if title and text:
                yield title, text
            el.clear()

def clean_wikitext(wikitext: str) -> str:
    """Снять разметку, оставить плоский текст."""
    code = mwparserfromhell.parse(wikitext)
    # удалить шаблоны и ссылки на файлы
    for tpl in code.filter_templates():
        try: code.remove(tpl)
        except Exception: pass
    text = code.strip_code(normalize=True, collapse=True)
    # Прибрать множественные пробелы и пустые строки
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def main():
    written = 0
    skipped = 0
    seen_names = set()
    pbar = tqdm(iter_pages(DUMP), desc="pages", unit="p")
    for title, wikitext in pbar:
        # пропускаем редиректы, disambiguation и т.п.
        if re.match(r"^\s*#REDIRECT", wikitext, re.IGNORECASE):
            skipped += 1; continue
        try:
            text = clean_wikitext(wikitext)
        except Exception:
            skipped += 1; continue
        if len(text) < MIN_LEN:
            skipped += 1; continue
        name = safe_name(title)
        # коллизии имён
        candidate = name
        i = 2
        while candidate in seen_names:
            candidate = f"{name}_{i}"; i += 1
        seen_names.add(candidate)
        (OUT / f"{candidate}.txt").write_text(text, encoding="utf-8")
        written += 1
        if written % 200 == 0:
            pbar.set_postfix(written=written, skipped=skipped)
    print(f"\nDone. Written: {written}, skipped: {skipped}. Out: {OUT}")

if __name__ == "__main__":
    main()
