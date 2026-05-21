"""
Строит «свой» лемматизатор для мальтийского из дампа Ġabra (MLRS UM).

Источник:
  resources/lexemes.bson    — леммы (21k записей)
  resources/wordforms.bson  — словоформы (~миллион записей)

Каждая wordform ссылается на лемму через `lexeme_id`. Из обоих файлов строим
плоский lookup-словарь {surface_form_lower: lemma}.

В случае коллизий (одна форма соответствует разным леммам) выбираем лемму
с максимальной частотой (norm_freq) или первую попавшуюся.

Выход:
  resources/mt_lemma_lookup.json  — {wordform: lemma}
"""
import json
import time
from pathlib import Path

import bson
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "resources"
OUT_JSON = RES / "mt_lemma_lookup.json"

def iter_bson(path: Path):
    """Потоково парсим BSON dump (один документ за раз)."""
    with open(path, "rb") as f:
        data = f.read()
    # bson.decode_file_iter не работает для concatenated dump; используем offset-парсинг
    offset = 0
    n = len(data)
    while offset < n:
        # Длина документа — первые 4 байта little-endian
        size = int.from_bytes(data[offset:offset+4], "little", signed=False)
        if size <= 0 or offset + size > n:
            break
        try:
            doc = bson.decode_all(data[offset:offset+size])[0]
        except Exception:
            offset += size; continue
        yield doc
        offset += size

def main():
    t0 = time.time()

    # 1) lexemes: _id → lemma
    print("[1/3] reading lexemes.bson ...")
    lexeme2info = {}
    for doc in tqdm(iter_bson(RES / "lexemes.bson"), desc="lexemes"):
        _id = doc.get("_id")
        lemma = doc.get("lemma")
        freq = doc.get("norm_freq", 0)
        if _id is not None and lemma:
            lexeme2info[_id] = (lemma, freq)
    print(f"  lexemes: {len(lexeme2info)}")

    # 2) wordforms: surface → lemma (через lexeme_id)
    print("[2/3] reading wordforms.bson ...")
    lookup = {}  # surface_lower → (lemma, freq)
    n_wf = 0
    n_no_lexeme = 0
    for doc in tqdm(iter_bson(RES / "wordforms.bson"), desc="wordforms"):
        n_wf += 1
        surf = doc.get("surface_form")
        lex_id = doc.get("lexeme_id")
        if not surf or lex_id is None:
            continue
        info = lexeme2info.get(lex_id)
        if not info:
            n_no_lexeme += 1; continue
        lemma, freq = info
        # нижний регистр для регистронечувствительного lookup
        key = surf.lower()
        prev = lookup.get(key)
        if prev is None or freq > prev[1]:
            lookup[key] = (lemma, freq)
    print(f"  wordforms total: {n_wf}, без lexeme: {n_no_lexeme}, в lookup: {len(lookup)}")

    # Также добавим сами леммы (форма == лемма)
    for lemma, freq in lexeme2info.values():
        key = lemma.lower()
        if key not in lookup or freq > lookup[key][1]:
            lookup[key] = (lemma, freq)

    # Доп. варианты без диакритик (часто встречаются в Wikipedia и web-текстах)
    DIA = str.maketrans({"ċ":"c", "ġ":"g", "ħ":"h", "ż":"z",
                         "Ċ":"C", "Ġ":"G", "Ħ":"H", "Ż":"Z"})
    extras_added = 0
    for key, (lemma, freq) in list(lookup.items()):
        bare = key.translate(DIA)
        if bare != key and bare not in lookup:
            lookup[bare] = (lemma, freq); extras_added += 1
    print(f"  diacritic-stripped variants added: {extras_added}")

    print(f"[3/3] final lookup size: {len(lookup)}")

    # Сохраним только {surface: lemma} без freq
    flat = {k: v[0] for k, v in lookup.items()}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False)
    print(f"Saved: {OUT_JSON} ({OUT_JSON.stat().st_size/1024/1024:.1f} MB)")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    # Демо
    print("\nDemo lookups:")
    for w in ["kitba", "ġurnata", "qieghed", "tieghi", "bnedem", "kitbu"]:
        print(f"  {w} -> {flat.get(w.lower(), '?')}")

if __name__ == "__main__":
    main()
