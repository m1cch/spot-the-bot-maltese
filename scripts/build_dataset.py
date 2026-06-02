"""
Строит датасет n-грамм из очищенного корпуса по методичке Spot the bot.

Алгоритм:
  для каждого текста (строка в input.txt):
    скользящее окно длины n (шаг 1)
    для каждого слова n-граммы:
        вектор[:m] из словаря (если слова нет — n-грамму пропускаем)
    конкатенация → вектор размерности n*m
    добавляем в датасет

Вход:
  --corpus  path/to/all_clean.txt      или склейку шардов
  --dict    embeddings/svd/mt_svd_k1024.npz
  -n        длина n-граммы (3-5)
  -m        размерность одного вектора (берём первые m компонент из SVD)

Выход:
  datasets/<name>__n{n}_m{m}.npy      — float32 матрица (N x n*m)
"""
import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "corpus_clean"
DICTS = ROOT / "embeddings" / "svd"
OUT = ROOT / "datasets"
OUT.mkdir(parents=True, exist_ok=True)


def load_dict(path: Path, m: int):
    """Возвращает (word→index, vectors[:m])."""
    data = np.load(path, allow_pickle=True)
    words = data["words"]
    vectors = data["vectors"][:, :m].astype(np.float32)
    idx = {w: i for i, w in enumerate(words)}
    return idx, vectors


def iter_corpus(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ws = line.strip().split()
            if ws: yield ws


def build_dataset(corpus_path: Path, idx, vectors, n: int, max_rows: int = None):
    m = vectors.shape[1]
    out_rows = []
    skipped = 0
    pbar = tqdm(iter_corpus(corpus_path), desc=f"n={n} m={m}")
    for words in pbar:
        if len(words) < n: continue
        for i in range(len(words) - n + 1):
            window = words[i:i+n]
            vecs = []
            ok = True
            for w in window:
                wi = idx.get(w)
                if wi is None:
                    ok = False; break
                vecs.append(vectors[wi])
            if not ok:
                skipped += 1; continue
            out_rows.append(np.concatenate(vecs))
            if max_rows and len(out_rows) >= max_rows:
                break
        if max_rows and len(out_rows) >= max_rows:
            break
    print(f"  built: {len(out_rows)}, skipped (OOV): {skipped}")
    return np.asarray(out_rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to all_clean.txt")
    ap.add_argument("--dict", required=True, help="path to mt_svd_k*.npz")
    ap.add_argument("-n", type=int, default=4, help="n-gram length")
    ap.add_argument("-m", type=int, default=8, help="word vector dim (slice of SVD vec)")
    ap.add_argument("--name", required=True, help="output name (e.g. human or bot)")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    print(f"=== n-gram dataset | n={args.n} m={args.m} name={args.name} ===")
    idx, vectors = load_dict(Path(args.dict), args.m)
    print(f"Vocab loaded: {len(idx)} words, dim={vectors.shape[1]}")

    data = build_dataset(Path(args.corpus), idx, vectors, args.n, args.max_rows)
    print(f"Final shape: {data.shape}")
    out = OUT / f"{args.name}__n{args.n}_m{args.m}.npy"
    np.save(out, data)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
