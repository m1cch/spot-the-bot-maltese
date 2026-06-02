"""
E3: FastText pre-trained word vectors для мальтийского.
По методичке: https://fasttext.cc/docs/en/crawl-vectors.html

URL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mt.300.bin.gz
Размер: ~1 GB (распакованный ~3.5 GB)

Скачиваем .bin.gz, распаковываем. Для интеграции с нашими скриптами
(словарь {word: vector}) — конвертируем в npz формат.
"""
import gzip
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "embeddings" / "fasttext"
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mt.300.bin.gz"


def download():
    import urllib.request
    out_gz = OUT / "cc.mt.300.bin.gz"
    out_bin = OUT / "cc.mt.300.bin"
    if out_bin.exists():
        print(f"already exists: {out_bin}")
        return out_bin
    if not out_gz.exists():
        print(f"downloading {URL} ...")
        with urllib.request.urlopen(URL) as r, open(out_gz, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            chunk = 1024 * 1024
            t0 = time.time()
            while True:
                buf = r.read(chunk)
                if not buf: break
                f.write(buf); done += len(buf)
                if total:
                    pct = 100*done/total
                    speed = done / 1024/1024 / max(time.time()-t0, 0.1)
                    print(f"\r  {done/1024/1024:7.1f}/{total/1024/1024:.1f} MB  {pct:5.1f}%  {speed:5.1f} MB/s",
                          end="", flush=True)
        print()
    print(f"unzipping ...")
    with gzip.open(out_gz, "rb") as g, open(out_bin, "wb") as f:
        shutil.copyfileobj(g, f)
    out_gz.unlink()
    print(f"unzipped: {out_bin}")
    return out_bin


def to_npz_subset(bin_path: Path, vocab_words: list, label: str):
    """Для каждого слова в vocab_words достаём FT-вектор и сохраняем npz."""
    import fasttext
    print(f"loading FT model from {bin_path} (this takes ~30s) ...")
    t0 = time.time()
    m = fasttext.load_model(str(bin_path))
    print(f"  loaded in {time.time()-t0:.1f}s")
    import numpy as np
    vectors = np.empty((len(vocab_words), 300), dtype=np.float32)
    for i, w in enumerate(vocab_words):
        vectors[i] = m.get_word_vector(w)
    words = np.array(vocab_words)
    out = OUT / f"mt_fasttext_{label}_d300.npz"
    np.savez_compressed(out, words=words, vectors=vectors)
    print(f"saved {out} ({out.stat().st_size/1024/1024:.1f} MB)")


def main():
    bin_path = download()
    if "--export-vocab" in sys.argv:
        # Вычитываем vocab из corpus_clean_L?
        idx = sys.argv.index("--export-vocab")
        corpus = Path(sys.argv[idx+1])
        label  = sys.argv[idx+2]
        words = set()
        for sp in sorted(corpus.glob("all_clean.shard*.txt")) + [corpus/"all_clean.txt"]:
            if not sp.exists(): continue
            with open(sp, "r", encoding="utf-8") as f:
                for line in f:
                    for w in line.split():
                        words.add(w)
        words = sorted(words)
        print(f"vocab from {corpus}: {len(words)} unique tokens")
        to_npz_subset(bin_path, words, label)


if __name__ == "__main__":
    main()
