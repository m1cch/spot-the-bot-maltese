"""
Сбор максимального количества мальтийской литературы из открытых источников.

Источники:
  1) archive.org — все элементы с language:maltese AND mediatype:texts
  2) Bible на мальтийском (ebible.org)
  3) Bloomlibrary — детские книги на mt
  4) Wikisource (общий, фильтр mt)

На выходе: corpus_raw/literature/<source>/<id>.txt
Один файл = один текст, имя файла = id источника.

Фильтр: пропускаем явный мусор (бюллетени Malta Amateur Radio League,
дубли с других языков, файлы < 500 символов).
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus_raw" / "literature"
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (spot-the-bot-mt research)"}

# Какие identifier'ы пропускаем как мусор
BLACKLIST_PATTERNS = [
    r"^marl_MARL",          # Malta Amateur Radio League бюллетени
    r"^arxiv-",             # arxiv (ошибочно помечены как мальтийские)
    r"Wandertipp",          # туристические гиды
    r"^http",               # tokopedia ссылки
    r"FIS[Ss]ongbook",      # song books
    r"Songbook",
    r"^mtwiki-",            # это уже в wiki, дубли
]
BL_RE = re.compile("|".join(BLACKLIST_PATTERNS), re.IGNORECASE)

def fetch_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def archive_org_list():
    """Получаем все 275+ identifier'ов мальтийских текстов."""
    print("== archive.org search ==")
    page = 1; rows = 100
    items = []
    while True:
        q = "language:(maltese) AND mediatype:(texts)"
        url = (f"https://archive.org/advancedsearch.php?"
               f"q={urllib.parse.quote(q)}&"
               f"fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&"
               f"rows={rows}&page={page}&output=json")
        d = fetch_json(url)
        docs = d["response"]["docs"]
        total = d["response"]["numFound"]
        if not docs: break
        items.extend(docs)
        print(f"  page {page}: +{len(docs)} (total fetched {len(items)}/{total})")
        if len(items) >= total: break
        page += 1
        time.sleep(0.5)
    return items

def archive_org_download(items, max_items: int = None):
    """Для каждого identifier: получить список файлов, скачать .txt (или .epub→.txt)."""
    import subprocess
    out_dir = OUT / "archive_org"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    ok = 0; skipped = 0
    for it in items:
        ident = it["identifier"]
        title = it.get("title", "")
        n += 1
        if max_items and ok >= max_items: break
        if BL_RE.search(ident) or BL_RE.search(title or ""):
            skipped += 1; continue
        # запрашиваем list files
        url_files = f"https://archive.org/metadata/{ident}/files"
        try:
            data = fetch_json(url_files)
        except Exception as e:
            print(f"  [!] {ident}: meta err {e}")
            continue
        files = data.get("result", []) if isinstance(data, dict) else []
        # ищем .txt — самый удобный формат
        txt_file = None
        epub_file = None
        for f in files:
            name = f.get("name", "")
            if name.endswith("_djvu.txt") or name.endswith(".txt"):
                if not txt_file or name.endswith("_djvu.txt"):
                    txt_file = name
            elif name.endswith(".epub"):
                epub_file = name
        chosen = txt_file or epub_file
        if not chosen:
            skipped += 1; continue
        # скачиваем
        target = out_dir / f"{ident}.txt"
        if target.exists() and target.stat().st_size > 500:
            ok += 1; continue
        durl = f"https://archive.org/download/{ident}/{urllib.parse.quote(chosen)}"
        try:
            content = fetch_bytes(durl, timeout=120)
        except Exception as e:
            print(f"  [!] {ident}: dl err {str(e)[:80]}")
            continue
        if chosen.endswith(".epub"):
            # извлечём txt из epub
            try:
                import zipfile, io, html
                z = zipfile.ZipFile(io.BytesIO(content))
                texts = []
                for inn in z.namelist():
                    if inn.lower().endswith((".xhtml", ".html", ".htm")):
                        raw = z.read(inn).decode("utf-8", errors="ignore")
                        # убрать теги
                        raw = re.sub(r"<[^>]+>", " ", raw)
                        raw = html.unescape(raw)
                        texts.append(raw)
                content = ("\n\n".join(texts)).encode("utf-8")
            except Exception as e:
                print(f"  [!] {ident}: epub parse err {str(e)[:80]}")
                continue
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
        # минимальный clean
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 500:
            skipped += 1; continue
        target.write_text(text, encoding="utf-8")
        ok += 1
        if ok % 20 == 0:
            print(f"  saved {ok} (scanned {n})")
        time.sleep(0.2)  # вежливо
    print(f"archive.org: scanned={n}, saved={ok}, skipped={skipped}")
    return ok

def fetch_bible_mt():
    """Качаем USFM/text Bible на мальтийском с ebible.org (CC BY-SA)."""
    print("\n== Bible (mt, ebible.org) ==")
    out_dir = OUT / "bible"
    out_dir.mkdir(parents=True, exist_ok=True)
    # ebible.org предоставляет ZIP с USFM-файлами. Для мальтийского:
    # https://ebible.org/Scriptures/mlt_readaloud.zip — Malta Bible (если есть)
    # https://ebible.org/find/details.php?id=mlt
    # Попробуем основные URL
    candidates = [
        "https://ebible.org/Scriptures/mlt_usfm.zip",
        "https://ebible.org/Scriptures/mlt-x-bible_usfm.zip",
        "https://ebible.org/Scriptures/maltese_usfm.zip",
    ]
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 200:
                    data = r.read()
                    target = out_dir / "mlt_usfm.zip"
                    target.write_bytes(data)
                    print(f"  saved {target} from {url}")
                    return target
        except Exception as e:
            print(f"  miss {url}: {str(e)[:60]}")
    print("  no Bible found at standard URLs")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-archive", type=int, default=None, help="limit archive items")
    ap.add_argument("--skip-archive", action="store_true")
    ap.add_argument("--skip-bible", action="store_true")
    args = ap.parse_args()

    if not args.skip_archive:
        items = archive_org_list()
        print(f"total archive.org hits: {len(items)}")
        archive_org_download(items, max_items=args.max_archive)

    if not args.skip_bible:
        fetch_bible_mt()

    print("\n== summary ==")
    for sub in sorted(OUT.glob("*")):
        if sub.is_dir():
            n = len(list(sub.glob("*")))
            print(f"  {sub.name}: {n} files")

if __name__ == "__main__":
    main()
