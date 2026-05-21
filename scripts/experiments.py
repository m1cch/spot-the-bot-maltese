"""
Шаг 7c — сравнение конфигураций: эмбеддинг (SVD / word2vec) × (n, m).

Свип Уишарта показал слабое расхождение human/bot при n=4,m=8,SVD.
Здесь для каждой конфигурации (тип эмбеддинга, длина n-граммы n, размерность
вектора слова m) строятся n-граммные облака человек/бот, кластеризуются
Уишартом и сравниваются — ищем конфигурацию с максимальным расхождением.

Запуск:
  python scripts/experiments.py

Выход:
  results/clustering/experiments_summary.json
"""
import json
import time
from pathlib import Path

import numpy as np

from build_dataset import load_dict, build_dataset
from wishart_compare import wishart, cluster_stats

ROOT = Path(__file__).resolve().parent.parent

SVD = ROOT / "embeddings" / "svd" / "mt_svd_k1024.npz"
W2V = ROOT / "embeddings" / "word2vec" / "mt_w2v_cbow_L2_d100.npz"
HUMAN = ROOT / "corpus_clean_L2" / "all_clean.shuf.txt"
BOT = ROOT / "corpus_bot" / "all_clean.shuf.txt"

CONFIGS = [
    ("svd", SVD, 3, 8),
    ("svd", SVD, 4, 8),
    ("svd", SVD, 5, 8),
    ("svd", SVD, 4, 16),
    ("svd", SVD, 4, 32),
    ("w2v", W2V, 4, 8),
]
K, H, SAMPLE, MAXROWS = 11, 1.0, 30000, 300000


def main():
    rng = np.random.default_rng(42)
    print(f"Wishart k={K} h={H} | sample={SAMPLE} per cloud\n")
    print(f"{'emb':>5} {'n':>2} {'m':>3} {'dim':>4} | {'H clu':>6} {'H noise':>8} {'H sil':>7}"
          f" | {'B clu':>6} {'B noise':>8} {'B sil':>7} | {'dClu':>5} {'dNoise':>7} {'dSil':>7}")
    print("-" * 104)
    results = []
    for emb, dictpath, n, m in CONFIGS:
        t0 = time.time()
        idx, vectors = load_dict(dictpath, m)
        st = {}
        for name, corpus in (("human", HUMAN), ("bot", BOT)):
            X = build_dataset(corpus, idx, vectors, n, MAXROWS)
            if len(X) > SAMPLE:
                X = X[rng.choice(len(X), SAMPLE, replace=False)]
            lab = wishart(X.astype(np.float32), K, H)
            st[name] = cluster_stats(X, lab)
        Hs, Bs = st["human"], st["bot"]
        results.append({"emb": emb, "n": n, "m": m, "dim": n * m,
                         "human": Hs, "bot": Bs})
        hsil = Hs.get("silhouette", float("nan"))
        bsil = Bs.get("silhouette", float("nan"))
        print(f"{emb:>5} {n:>2} {m:>3} {n*m:>4} | "
              f"{Hs['n_clusters']:>6} {Hs['noise_fraction']*100:>7.1f}% {hsil:>7.3f} | "
              f"{Bs['n_clusters']:>6} {Bs['noise_fraction']*100:>7.1f}% {bsil:>7.3f} | "
              f"{Bs['n_clusters']-Hs['n_clusters']:>5} "
              f"{(Bs['noise_fraction']-Hs['noise_fraction'])*100:>6.1f}% "
              f"{(Bs.get('silhouette',0)-Hs.get('silhouette',0)):>7.3f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "results" / "clustering" / "experiments_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": {"k": K, "h": H, "sample": SAMPLE},
                               "configs": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
