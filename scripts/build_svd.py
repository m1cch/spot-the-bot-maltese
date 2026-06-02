"""
Строит SVD-словарь по очищенному мальтийскому корпусу.

По методичке Spot the bot:
  1) TfidfVectorizer(token_pattern=<мальтийский алфавит + цифры>) строит
     матрицу A (n_terms x n_docs).
  2) Усечённое SVD: A ≈ U * Σ * V^T. Берём строки U * Σ как словесные векторы.
  3) Сохраняем dict {word: vector[k]} в .npy/.json.

Преимущество SVD из методички:
  для вектора размерности d < k достаточно взять первые d компонент,
  пересчитывать не надо.

Вход:
  corpus_clean/all_clean.txt              — единый файл (строка = очищенный текст).
  ИЛИ:
  corpus_clean/all_clean.shard*.txt       — если уже шардированный.

Выход:
  embeddings/svd/mt_svd_k{K}.npy          — np.savez_compressed(words, vectors)
  embeddings/svd/mt_svd_k{K}.meta.json    — мета-инфа
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "corpus_clean"
OUT = ROOT / "embeddings" / "svd"
OUT.mkdir(parents=True, exist_ok=True)

# Мальтийский алфавит + цифры (для TfidfVectorizer)
# Мальтийский: a b ċ d e f ġ g għ h ħ i ie j k l m n o p q r s t u v w x ż z
# token_pattern берёт ТОЛЬКО эти буквы (нет пунктуации)
MT_ALPHABET = r"a-zA-ZċġħżĊĠĦŻ"
TOKEN_PATTERN = rf"[{MT_ALPHABET}0-9\-']+"


def load_corpus():
    """Объединить все шарды (если есть) либо взять одиночный all_clean.txt."""
    shards = sorted(CLEAN.glob("all_clean.shard*.txt"))
    if shards:
        print(f"Found {len(shards)} shard files")
        out = []
        for sp in shards:
            with open(sp, "r", encoding="utf-8") as f:
                out.extend(line.strip() for line in f if line.strip())
        return out
    single = CLEAN / "all_clean.txt"
    if not single.exists():
        raise FileNotFoundError(f"No corpus_clean files found in {CLEAN}")
    with open(single, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--rank", type=int, default=1024, help="SVD rank")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.6)
    ap.add_argument("--max-features", type=int, default=None,
                    help="vocab cap (None=no cap)")
    args = ap.parse_args()

    print(f"=== SVD-словарь | k={args.rank} ===")
    t0 = time.time()
    docs = load_corpus()
    print(f"Documents: {len(docs)}")
    print(f"Avg length: {sum(len(d.split()) for d in docs)/len(docs):.0f} tokens")

    print(f"\n[1/3] TfidfVectorizer (token_pattern={TOKEN_PATTERN!r})")
    vec = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        dtype=np.float32,
    )
    A = vec.fit_transform(docs)  # (n_docs x n_terms), CSR
    A = A.T.tocsr()              # (n_terms x n_docs)
    words = vec.get_feature_names_out()
    n_terms, n_docs = A.shape
    print(f"  shape A: {A.shape} (terms x docs)")
    print(f"  vocab size: {n_terms}")
    print(f"  matrix density: {A.nnz/(n_terms*n_docs):.6f}")

    k = min(args.rank, min(A.shape) - 1)
    if k != args.rank:
        print(f"  rank capped to {k} (matrix is small)")
    print(f"\n[2/3] Truncated SVD (k={k})")

    # scipy.sparse.linalg.svds — возвращает в возрастающем порядке σ
    U, s, Vt = svds(A.astype(np.float32), k=k)

    # отсортируем по убыванию σ
    order = np.argsort(-s)
    U = U[:, order]
    s = s[order]
    print(f"  U: {U.shape}, σ[:5]: {s[:5]}")
    print(f"  σ[-5:]: {s[-5:]}")

    # Векторы слов = U * Σ (n_terms x k)
    vectors = (U * s).astype(np.float32)
    print(f"  vectors shape: {vectors.shape}")

    out_path = OUT / f"mt_svd_k{k}.npz"
    np.savez_compressed(out_path, words=words, vectors=vectors)
    meta = {
        "rank": k,
        "n_docs": int(n_docs),
        "n_terms": int(n_terms),
        "min_df": args.min_df,
        "max_df": args.max_df,
        "token_pattern": TOKEN_PATTERN,
        "elapsed_sec": time.time() - t0,
        "singular_values_head": s[:10].tolist(),
        "singular_values_tail": s[-10:].tolist(),
    }
    (OUT / f"mt_svd_k{k}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[3/3] Saved: {out_path}")
    print(f"Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
