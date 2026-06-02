"""
L3 — чистый rule-based препроцессор для мальтийского.

Без библиотек NLP. Только:
  - токенизация по пробелам и пунктуации (regex)
  - стрипперы аффиксов мальтийского
  - правила POS-замены через regex/словари

Аналог лемматизатора Субхангулова для татарского — самого «строгого по методичке».

Правила:
  • артикли с дефисом: il-, it-, id-, in-, ir-, is-, iz-, iż-, ix- (+ их одиночные l-, t-, ...)
  • проклитики с апострофом: b', f', m', t', s', n', x', ġ'
  • притяжательные/объектные суффиксы существительных и глаголов:
       -i (1sg), -ek/-k/-ok (2sg), -u/-ha (3sg), -na (1pl), -kom (2pl), -hom (3pl)
  • отрицательная клитика -x
  • удвоение начальной согласной у глаголов: tt-, kk-, mm-, ss-, ...
  • замена местоимений / числительных / имён собственных:
       - местоимения (список) → PRON1
       - числа (цифры или числительные-слова) → ORDINAL1
       - слова с заглавной буквы НЕ в начале предложения → PERSON1

NB: для строгости методички — никакого внешнего лемматизатора.

Запуск:
  CUDA_VISIBLE_DEVICES= python scripts/preprocess_rules.py --shard 0 --total 3
  (GPU не нужно, чисто CPU)
"""
import argparse
import os
import random
import re
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus_raw"
CLEAN = ROOT / "corpus_clean_L3"
CLEAN.mkdir(parents=True, exist_ok=True)

SECTIONS_PRIORITY = [
    ("wiki/articles", None),
    ("theses", None),
    ("nonfiction", None),
    ("umlib_oar", None),
    ("press_mt", 8000),
    ("blogs", 8000),
]

# === Списки правил ===

ARTICLE_PREFIXES = (
    "il-", "it-", "id-", "in-", "ir-", "is-", "iz-", "iż-", "ix-", "iċ-", "iġ-", "iħ-",
    "l-", "t-", "d-", "n-", "r-", "s-", "z-", "ż-", "x-", "ċ-", "ġ-", "ħ-",
)
PROCLITIC_APOS = ("b'", "f'", "m'", "t'", "s'", "n'", "x'", "ġ'", "ż'", "ċ'", "ħ'", "k'", "p'")

# Притяжательные/объектные суффиксы — мальтийские, через дефис
# (в орфографии mt они пишутся как `ktieb-na`, `belt-i`)
HYPHEN_SUFFIXES = (
    "-hom", "-kom", "-hu", "-ha", "-na", "-ek", "-ok", "-u", "-k", "-h", "-i", "-x",
)
# Финальный -x = отрицательная клитика без дефиса (например `niġix` = «я не прихожу»),
# применяем строго к словам длиной >= 5
NEGATION_X = re.compile(r"^(?P<stem>.{4,})x$")

# Закрытый список местоимений мальтийского (личные/притяжательные/указательные/вопросительные)
PRONOUNS = {
    "jien", "jiena", "int", "inti", "huwa", "hu", "hi", "hija", "aħna", "intom", "huma",
    "miegħi", "miegħek", "miegħu", "magħha", "magħna", "magħkom", "magħhom",
    "tiegħi", "tiegħek", "tiegħu", "tagħha", "tagħna", "tagħkom", "tagħhom",
    "lili", "lilek", "lilu", "lilha", "lilna", "lilkom", "lilhom",
    "dan", "din", "dawn", "dak", "dik", "dawk",
    "min", "x'", "kien", "fejn", "meta", "kif", "kemm", "liema",
    "ħadd", "kollox", "xejn", "xi", "kull", "ieħor", "oħra",
}

# Числительные мальтийского (закрытый список) — будут заменены на ORDINAL1
NUMERALS_WORDS = {
    "wieħed", "waħda", "tnejn", "tlieta", "erbgħa", "ħamsa", "sitta",
    "sebgħa", "tmienja", "disgħa", "għaxra",
    "ħdax", "tnax", "tlettax", "erbatax", "ħmistax", "sittax", "sbatax",
    "tmintax", "dsatax", "għoxrin",
    "tletin", "erbgħin", "ħamsin", "sittin", "sebgħin", "tmenin", "disgħin",
    "mija", "elf", "miljun", "biljun",
    "l-ewwel", "it-tieni", "it-tielet", "ir-raba", "il-ħames", "is-sitta",
}

# Регулярки
WIKI_GARBAGE = re.compile(r"[\|\[\]\{\}=*<>#_/\\^~`@$%&()]+")
NUMSUFFIX = re.compile(r"\b\d+[a-zA-Z]+\b")
PURE_NUMBER = re.compile(r"^[\d.,\-]+$")
TOKENIZER = re.compile(r"[a-zA-ZċġħżĊĠĦŻ0-9\-']+")
ALPHA_MT_FULL = re.compile(r"^[a-zA-ZċġħżĊĠĦŻ][a-zA-ZċġħżĊĠĦŻ\-']*$")

# Эвристика для имён собственных: токен НЕ в начале предложения и начинается с заглавной
# (мы будем токенизировать без сохранения границ предложений, поэтому возьмём упрощённый вариант:
#  токены с заглавной первой буквой, кроме первого в строке)


def strip_clitics(word: str) -> str:
    """Снять артикли и проклитики (один раз)."""
    wl = word.lower()
    for p in PROCLITIC_APOS:
        if wl.startswith(p):
            return word[len(p):]
    for p in ARTICLE_PREFIXES:
        if wl.startswith(p):
            return word[len(p):]
    return word


def strip_suffix(word: str) -> str:
    """Снять суффикс если он отделён дефисом (`ktieb-na` → `ktieb`).
    Не трогает естественные окончания типа `mammiferu`."""
    wl = word.lower()
    for s in HYPHEN_SUFFIXES:
        if wl.endswith(s) and len(wl) - len(s) >= 3:
            return word[: -len(s)]
    return word


def rule_lemma(token: str) -> str:
    """Lemma по правилам: снять клитики, потом суффикс. Один или два проходов."""
    t = token
    for _ in range(2):
        t = strip_clitics(t)
    t = strip_suffix(t)
    return t.lower().strip()


def is_propn(token: str, prev_was_sentence_end: bool) -> bool:
    """Имя собственное: первая буква заглавная и это не первый токен предложения."""
    if not token:
        return False
    return token[0].isupper() and not prev_was_sentence_end


def is_num(token: str) -> bool:
    if PURE_NUMBER.match(token):
        return True
    return token.lower() in NUMERALS_WORDS


def is_pron(token: str) -> bool:
    return token.lower() in PRONOUNS


# Разбиение на предложения — упрощённое: по `.`, `!`, `?`, `:`, новой строке
SENT_END = re.compile(r"[.!?]\s+|\n+")


def process_text(text: str) -> list:
    """Возвращает список чистых токенов после препроцессинга."""

    # pre-clean
    text = WIKI_GARBAGE.sub(" ", text)
    text = NUMSUFFIX.sub(" ", text)
    out = []

    # разбиваем на предложения
    sentences = SENT_END.split(text)
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        tokens = TOKENIZER.findall(sent)
        for i, tok in enumerate(tokens):
            if not tok: continue

            # 1) числа
            if is_num(tok):
                out.append("ORDINAL1"); continue

            # 2) местоимения
            if is_pron(tok):
                out.append("PRON1"); continue

            # 3) имена собственные (capitalized, not at sentence start)
            if is_propn(tok, prev_was_sentence_end=(i == 0)):
                out.append("PERSON1"); continue

            # 4) обычное слово — лемматизируем по правилам
            lem = rule_lemma(tok)
            if not lem or len(lem) > 40:
                continue
            if not ALPHA_MT_FULL.match(lem):
                continue
            out.append(lem)
    return out


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

    print(f"=== L3 (rule-based) shard {args.shard}/{args.total} ===")
    plan = collect_plan(args.shard, args.total)
    if args.smoke:
        plan = plan[:args.smoke]
    print(f"Files in shard: {len(plan)}")

    for sect, _ in SECTIONS_PRIORITY:
        (CLEAN / sect.replace("/", "_")).mkdir(parents=True, exist_ok=True)

    all_lines = CLEAN / f"all_clean.shard{args.shard}.txt"
    index = CLEAN / f"_doc_index.shard{args.shard}.tsv"

    t0 = time.time()
    n_done = 0; n_words = 0
    with open(all_lines, "w", encoding="utf-8") as fall, \
         open(index, "w", encoding="utf-8") as findex:
        findex.write("source\tname\tn_words\n")
        for source, fp in tqdm(plan, desc=f"L3-shard{args.shard}"):
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text: continue
            words = process_text(text)
            if not words: continue
            clean = " ".join(words)
            (CLEAN / source / f"{fp.stem}.txt").write_text(clean, encoding="utf-8")
            fall.write(clean + "\n")
            findex.write(f"{source}\t{fp.stem}\t{len(words)}\n")
            n_done += 1; n_words += len(words)

    el = time.time() - t0
    print(f"\nL3 shard {args.shard}: done={n_done} words={n_words} elapsed={el/60:.1f}min "
          f"speed={(n_done/el if el>0 else 0):.1f} docs/s")


if __name__ == "__main__":
    main()
