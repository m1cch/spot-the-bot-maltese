"""
E2: Word2Vec (CBOW и Skipgram) на очищенном мальтийском корпусе.

По методичке Spot the bot — ноутбук word2vec.ipynb С. Суровой.
Используем gensim, поскольку он стандартный для академических работ.

Аргументы:
  --corpus  corpus_clean_L1/all_clean.shard*.txt (или единый all_clean.txt)
  --label   L1 | L2 | L3   (для имени выходного файла)
  --dim     размерность (по умолчанию 100)
  --window  контекстное окно (по умолчанию 5)
  --min-count  минимальная частота слова (по умолчанию 5)

Выход:
  embeddings/word2vec/mt_w2v_cbow_{label}_d{dim}.npz
  embeddings/word2vec/mt_w2v_sg_{label}_d{dim}.npz

Формат — совместим с build_dataset.py:
  npz: words (array), vectors (n_words x dim)
"""
import argparse
import time
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parent.parent

def load_corpus(corpus_dir: Path):
    """Считать all_clean.shard*.txt или одиночный all_clean.txt — list[list[str]]."""
    shards = sorted(corpus_dir.glob("all_clean.shard*.txt"))
    if not shards:
        shards = [corpus_dir / "all_clean.txt"]
    sentences = []
    for sp in shards:
        if not sp.exists(): continue
        with open(sp, "r", encoding="utf-8") as f:
            for line in f:
                ws = line.strip().split()
                if ws: sentences.append(ws)
    return sentences

def train_save(sents, sg: int, args, out_path: Path):
    print(f"\n[train sg={sg}] dim={args.dim} window={args.window} min_count={args.min_count}")
    t0 = time.time()
    model = Word2Vec(
        sentences=sents,
        vector_size=args.dim,
        window=args.window,
        min_count=args.min_count,
        sg=sg,
        workers=args.workers,
        epochs=args.epochs,
    )
    print(f"  trained in {time.time()-t0:.1f}s | vocab={len(model.wv)}")
    words = np.array(model.wv.index_to_key)
    vectors = model.wv.vectors.astype(np.float32)
    np.savez_compressed(out_path, words=words, vectors=vectors)
    print(f"  saved {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to corpus_clean_L?/")
    ap.add_argument("--label", required=True, help="L1|L2|L3")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    out_dir = ROOT / "embeddings" / "word2vec"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Word2Vec | label={args.label} | corpus={corpus_dir} ===")
    t0 = time.time()
    sents = load_corpus(corpus_dir)
    print(f"sentences: {len(sents)} | total tokens: {sum(len(s) for s in sents)}")

    # CBOW
    train_save(sents, sg=0, args=args,
               out_path=out_dir / f"mt_w2v_cbow_{args.label}_d{args.dim}.npz")
    # Skipgram
    train_save(sents, sg=1, args=args,
               out_path=out_dir / f"mt_w2v_sg_{args.label}_d{args.dim}.npz")

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
