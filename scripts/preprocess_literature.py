"""
Допроцессинг literature/archive_org/ во все три варианта препроцессинга (L1/L2/L3).

Запускается ПОСЛЕ окончания L1/L2/L3 — добавляет архивные тексты как ещё одну
секцию `literature_archive_org` в каждый corpus_clean_L?/ и дописывает в
all_clean.shard0.txt (мы используем shard 0 как «overflow» для пост-обработки).

Использование:
  python scripts/preprocess_literature.py --variant L1
  python scripts/preprocess_literature.py --variant L2
  python scripts/preprocess_literature.py --variant L3
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "corpus_raw" / "literature" / "archive_org"
RES = ROOT / "resources"

# ----- общие константы (копируем из L1/L2/L3 для согласованности) -----
POS_REPLACE = {"PROPN": "PERSON1", "PRON": "PRON1", "NUM": "ORDINAL1"}
POS_DROP = {"PUNCT", "SYM", "X", "SPACE"}
ALPHA_MT_FULL = re.compile(r"^[a-zA-ZċġħżĊĠĦŻ][a-zA-ZċġħżĊĠĦŻ\-']*$")
WIKI_GARBAGE = re.compile(r"[\|\[\]\{\}=*<>#_/\\^~`@$%&]+")
NUMSUFFIX = re.compile(r"\b\d+[a-zA-Z]+\b")
SENT_END = re.compile(r"(?<=[.!?:])\s+")
MAX_SENT_CHARS = 1500
DIA = str.maketrans({"ċ":"c","ġ":"g","ħ":"h","ż":"z","Ċ":"C","Ġ":"G","Ħ":"H","Ż":"Z"})
ARTICLE_PREFIXES = ("il-","it-","id-","in-","ir-","is-","iz-","iż-","ix-","iċ-","iġ-","iħ-",
                    "l-","t-","d-","n-","r-","s-","z-","ż-","x-","ċ-","ġ-","ħ-")
PROCLITIC_APOS = ("b'","f'","m'","t'","s'","n'","x'","ġ'","ż'","ċ'","ħ'","k'","p'")
HYPHEN_SUFFIXES = ("-hom","-kom","-hu","-ha","-na","-ek","-ok","-u","-k","-h","-i","-x")

PRONOUNS = {
    "jien","jiena","int","inti","huwa","hu","hi","hija","aħna","intom","huma",
    "miegħi","miegħek","miegħu","magħha","magħna","magħkom","magħhom",
    "tiegħi","tiegħek","tiegħu","tagħha","tagħna","tagħkom","tagħhom",
    "lili","lilek","lilu","lilha","lilna","lilkom","lilhom",
    "dan","din","dawn","dak","dik","dawk","min","fejn","meta","kif","kemm","liema",
}
NUMERALS_WORDS = {
    "wieħed","waħda","tnejn","tlieta","erbgħa","ħamsa","sitta",
    "sebgħa","tmienja","disgħa","għaxra","ħdax","tnax","tlettax","erbatax","ħmistax",
    "sittax","sbatax","tmintax","dsatax","għoxrin","tletin","erbgħin","ħamsin",
    "sittin","sebgħin","tmenin","disgħin","mija","elf","miljun","biljun",
}
TOKENIZER = re.compile(r"[a-zA-ZċġħżĊĠĦŻ0-9\-']+")
PURE_NUMBER = re.compile(r"^[\d.,\-]+$")

# ----- helpers (общие) -----


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


def normalize_lemma(lemma: str) -> str:
    if not lemma: return ""
    lemma = lemma.lower().strip()
    if not lemma or len(lemma) > 40: return ""
    if not ALPHA_MT_FULL.match(lemma): return ""
    return lemma

# ----- L1: Stanza + neural lemma -----


def process_L1(files, batch_docs=16):
    import stanza
    nlp = stanza.Pipeline(lang="mt", processors="tokenize,pos,lemma",
                          use_gpu=True, verbose=False, download_method=None,
                          tokenize_batch_size=128, pos_batch_size=128, lemma_batch_size=64)
    out_dir = ROOT / "corpus_clean_L1" / "literature_archive_org"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_lines_path = ROOT / "corpus_clean_L1" / "all_clean.literature.txt"
    return _run_stanza(nlp, files, out_dir, all_lines_path, batch_docs,
                       lemma_fn=lambda txt, lm: (lm or txt).lower())

# ----- L2: Stanza + Ġabra lookup -----


def process_L2(files, batch_docs=16):
    import stanza
    nlp = stanza.Pipeline(lang="mt", processors="tokenize,pos,lemma",
                          use_gpu=True, verbose=False, download_method=None,
                          tokenize_batch_size=128, pos_batch_size=128, lemma_batch_size=64)
    with open(RES/"mt_lemma_lookup.json","r",encoding="utf-8") as f:
        LK = json.load(f)

    def strip_clitics(w):
        wl = w.lower()
        for p in PROCLITIC_APOS:
            if wl.startswith(p): return w[len(p):], True
        for p in ARTICLE_PREFIXES:
            if wl.startswith(p): return w[len(p):], True
        return w, False

    def custom_lemma(surface, stanza_lemma):
        s = (surface or "").lower().strip()
        if not s: return ""
        if s in LK: return LK[s]
        s_nd = s.translate(DIA)
        if s_nd in LK: return LK[s_nd]
        stripped, changed = strip_clitics(s)
        if changed and stripped in LK: return LK[stripped]
        if changed and stripped.translate(DIA) in LK: return LK[stripped.translate(DIA)]
        if stanza_lemma:
            sl = stanza_lemma.lower().strip()
            if sl: return sl
        return s
    out_dir = ROOT / "corpus_clean_L2" / "literature_archive_org"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_lines_path = ROOT / "corpus_clean_L2" / "all_clean.literature.txt"
    return _run_stanza(nlp, files, out_dir, all_lines_path, batch_docs,
                       lemma_fn=custom_lemma)


def _run_stanza(nlp, files, out_dir, all_lines_path, batch_docs, lemma_fn):
    import stanza
    fall = open(all_lines_path, "w", encoding="utf-8")
    n_done = 0; n_words = 0
    pbar = tqdm(total=len(files), desc=out_dir.parent.name)

    for i in range(0, len(files), batch_docs):
        batch = files[i:i+batch_docs]

        # собираем чанки по всем файлам батча
        chunks_per_file, flat_texts = [], []
        for fp in batch:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                chunks_per_file.append((fp, 0)); continue
            chs = split_to_chunks(text)
            chunks_per_file.append((fp, len(chs)))
            flat_texts.extend(chs)
        if not flat_texts:
            pbar.update(len(batch)); continue

        # прогоняем весь батч через stanza одним вызовом
        try:
            docs_in = [stanza.Document([], text=t) for t in flat_texts]
            docs_out = nlp(docs_in)
        except Exception as e:
            print(f"\n[!] batch failed: {e}"); pbar.update(len(batch)); continue

        # раскладываем результат обратно по исходным файлам
        cursor = 0
        for fp, n_chunks in chunks_per_file:
            if n_chunks == 0: pbar.update(1); continue
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
                        lem = lemma_fn(w.text, w.lemma)
                        tok = normalize_lemma(lem)
                        if tok: words.append(tok)
            if not words: pbar.update(1); continue
            clean = " ".join(words)
            (out_dir / f"{fp.stem}.txt").write_text(clean, encoding="utf-8")
            fall.write(clean + "\n")
            n_done += 1; n_words += len(words); pbar.update(1)
    pbar.close(); fall.close()
    return n_done, n_words

# ----- L3: rule-based -----


def process_L3(files):
    def strip_clitics(w):
        wl = w.lower()
        for p in PROCLITIC_APOS:
            if wl.startswith(p): return w[len(p):]
        for p in ARTICLE_PREFIXES:
            if wl.startswith(p): return w[len(p):]
        return w

    def strip_suffix(w):
        wl = w.lower()
        for s in HYPHEN_SUFFIXES:
            if wl.endswith(s) and len(wl)-len(s) >= 3:
                return w[:-len(s)]
        return w

    def rule_lemma(tok):
        t = tok
        for _ in range(2): t = strip_clitics(t)
        t = strip_suffix(t)
        return t.lower().strip()
    out_dir = ROOT / "corpus_clean_L3" / "literature_archive_org"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_lines_path = ROOT / "corpus_clean_L3" / "all_clean.literature.txt"
    fall = open(all_lines_path, "w", encoding="utf-8")
    n_done = 0; n_words = 0
    for fp in tqdm(files, desc="L3 literature"):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception: continue
        if not text: continue
        text = WIKI_GARBAGE.sub(" ", text)
        text = NUMSUFFIX.sub(" ", text)
        out = []
        for sent in SENT_END.split(text):
            sent = sent.strip()
            if not sent: continue
            tokens = TOKENIZER.findall(sent)
            for i, tok in enumerate(tokens):
                if not tok: continue
                if PURE_NUMBER.match(tok) or tok.lower() in NUMERALS_WORDS:
                    out.append("ORDINAL1"); continue
                if tok.lower() in PRONOUNS:
                    out.append("PRON1"); continue
                if tok[0].isupper() and i > 0:
                    out.append("PERSON1"); continue
                lem = rule_lemma(tok)
                if not lem or len(lem)>40: continue
                if not ALPHA_MT_FULL.match(lem): continue
                out.append(lem)
        if not out: continue
        clean = " ".join(out)
        (out_dir / f"{fp.stem}.txt").write_text(clean, encoding="utf-8")
        fall.write(clean + "\n")
        n_done += 1; n_words += len(out)
    fall.close()
    return n_done, n_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["L1","L2","L3"])
    args = ap.parse_args()
    files = sorted(SOURCE_DIR.glob("*.txt"))
    print(f"=== literature → {args.variant} ({len(files)} files) ===")
    t0 = time.time()
    if args.variant == "L1":
        n, w = process_L1(files)
    elif args.variant == "L2":
        n, w = process_L2(files)
    else:
        n, w = process_L3(files)
    print(f"\n{args.variant}: done={n} words={w} elapsed={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
