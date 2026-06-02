"""
Шаг 7 — кластеризация Уишарта облаков «человек» и «бот» + сравнение.

По методичке Spot the bot: n-граммные облака человеческого и бот-корпуса
кластеризуются алгоритмом Уишарта (Wishart 1969, mode analysis — плотностная
кластеризация), затем сравнивается структура кластеров и метрики качества
(глава 23 Aggarwal & Reddy — внутренние меры валидности).

Алгоритм Уишарта (реализация ниже):
  1. Оценка плотности каждой точки по расстоянию до k-го соседа:
       p_i = k / (n * V_d * r_i^d),  V_d — объём единичного d-шара.
  2. Точки обрабатываются в порядке убывания плотности.
  3. Новая точка, по её kNN-связям с уже обработанными:
       - нет связей           -> новый кластер;
       - один кластер         -> присоединяется (если тот не «завершён»);
       - несколько кластеров  -> значимые (perepad плотности >= h) замораживаются,
                                 точка уходит в шум; незначимые сливаются.
  «Значимость» кластера: max(p) - min(p) по его точкам >= h.

Вход:
  --human  datasets/human__n*_m*.npy
  --bot    datasets/bot__n*_m*.npy

Выход:
  results/clustering/wishart_report.json   — метрики и сравнение
  results/clustering/wishart_<name>.png    — гистограммы размеров кластеров

Запуск:
  python scripts/wishart_compare.py --human datasets/human__n4_m8.npy \
         --bot datasets/bot__n4_m8.npy -k 11 --hh 0.2 --sample 30000
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "clustering"
OUT.mkdir(parents=True, exist_ok=True)


def wishart(X: np.ndarray, k: int, h: float):
    """Кластеризация Уишарта. Возвращает метки (0 = шум/фон)."""
    n, d = X.shape
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dist, idx = nn.kneighbors(X)               # dist[:,0]=0 (сама точка)
    r_k = dist[:, k]                            # расстояние до k-го соседа
    V_d = math.pi ** (d / 2) / gamma(d / 2 + 1)  # объём единичного d-шара
    p = k / (n * V_d * np.power(r_k, d) + 1e-300)  # плотность

    order = np.argsort(-p)                      # убывание плотности
    labels = np.full(n, -1, dtype=np.int64)     # -1 = ещё не обработана
    completed = {0: True}                       # кластер 0 = шум, всегда «завершён»
    c_min, c_max = {}, {}                       # экстремумы плотности по кластеру
    next_c = 0

    def significant(c):
        return c != 0 and c in c_min and (c_max[c] - c_min[c]) >= h

    for i in order:
        neigh = idx[i, 1:k + 1]
        lab = labels[neigh]
        uniq = set(int(x) for x in lab[lab != -1])

        if not uniq:                             # — изолированная точка
            next_c += 1
            labels[i] = next_c
            completed[next_c] = False
            c_min[next_c] = c_max[next_c] = p[i]

        elif len(uniq) == 1:                     # — один кластер
            c = next(iter(uniq))
            if completed.get(c, False):
                labels[i] = 0
            else:
                labels[i] = c
                c_min[c] = min(c_min[c], p[i])
                c_max[c] = max(c_max[c], p[i])

        else:                                    # — несколько кластеров
            if all(completed.get(c, False) for c in uniq):
                labels[i] = 0
            else:
                sig = [c for c in uniq if significant(c)]
                if sig:                          # есть значимый — точка на «перевале»
                    for c in uniq:
                        if c == 0 or completed.get(c, False):
                            continue
                        if significant(c):
                            completed[c] = True          # заморозить значимый
                        else:
                            labels[labels == c] = 0      # незначимый -> шум
                            c_min.pop(c, None); c_max.pop(c, None)
                            completed.pop(c, None)
                    labels[i] = 0
                else:                            # все незначимые — слить
                    merge = [c for c in uniq if c != 0 and not completed.get(c, False)]
                    tgt = min(merge)
                    for c in merge:
                        if c != tgt:
                            labels[labels == c] = tgt
                            c_min[tgt] = min(c_min[tgt], c_min.pop(c))
                            c_max[tgt] = max(c_max[tgt], c_max.pop(c))
                            completed.pop(c, None)
                    labels[i] = tgt
                    c_min[tgt] = min(c_min[tgt], p[i])
                    c_max[tgt] = max(c_max[tgt], p[i])

    return labels


def cluster_stats(X, labels):
    """Сводка по кластеризации + внутренние метрики качества (гл. 23 A&R)."""
    uniq, counts = np.unique(labels, return_counts=True)
    sizes = {int(u): int(c) for u, c in zip(uniq, counts)}
    n_noise = sizes.get(0, 0)
    real = labels != 0
    n_clusters = len([u for u in uniq if u != 0])

    stats = {
        "n_points": int(len(labels)),
        "n_clusters": n_clusters,
        "noise_points": n_noise,
        "noise_fraction": float(n_noise / len(labels)),
        "cluster_sizes": sorted((v for k, v in sizes.items() if k != 0), reverse=True)[:20],
        "largest_cluster_frac": float(max((v for k, v in sizes.items() if k != 0),
                                          default=0) / len(labels)),
    }

    # внутренние меры валидности — только если есть >=2 непустых кластера среди не-шумовых
    if n_clusters >= 2 and real.sum() > n_clusters:
        Xr, Lr = X[real], labels[real]
        if len(np.unique(Lr)) >= 2:
            sample = np.random.choice(len(Xr), min(5000, len(Xr)), replace=False)
            stats["silhouette"] = float(silhouette_score(Xr[sample], Lr[sample]))
            stats["davies_bouldin"] = float(davies_bouldin_score(Xr, Lr))
            stats["calinski_harabasz"] = float(calinski_harabasz_score(Xr, Lr))
    return stats


def plot_sizes(stats_h, stats_b):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib нет — график пропущен)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, st, name in ((axes[0], stats_h, "human"), (axes[1], stats_b, "bot")):
        sizes = st["cluster_sizes"]
        ax.bar(range(len(sizes)), sizes)
        ax.set_title(f"{name}: {st['n_clusters']} кластеров, "
                     f"шум {st['noise_fraction']*100:.1f}%")
        ax.set_xlabel("кластер (по убыванию размера)")
        ax.set_ylabel("размер")
    fig.tight_layout()
    fig.savefig(OUT / "wishart_sizes.png", dpi=120)
    print(f"  график: {OUT/'wishart_sizes.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="datasets/human__n*_m*.npy")
    ap.add_argument("--bot", required=True, help="datasets/bot__n*_m*.npy")
    ap.add_argument("-k", "--neighbors", type=int, default=11, help="k для Уишарта")
    ap.add_argument("--hh", type=float, default=0.2, help="порог значимости h")
    ap.add_argument("--sample", type=int, default=30000,
                    help="подвыборка n-грамм на облако (0 = всё)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    print(f"=== Wishart compare | k={args.neighbors} h={args.hh} ===")
    report = {"params": {"k": args.neighbors, "h": args.hh, "sample": args.sample}}

    for name, path in (("human", args.human), ("bot", args.bot)):
        X = np.load(path).astype(np.float32)
        if args.sample and len(X) > args.sample:
            sel = np.random.choice(len(X), args.sample, replace=False)
            X = X[sel]
        print(f"\n[{name}] {path} -> cloud {X.shape}")
        t0 = time.time()
        labels = wishart(X, args.neighbors, args.hh)
        st = cluster_stats(X, labels)
        st["elapsed_sec"] = time.time() - t0
        report[name] = st
        print(f"  clusters={st['n_clusters']} noise={st['noise_fraction']*100:.1f}% "
              f"| {st.get('silhouette', float('nan')):.3f} sil "
              f"| {st['elapsed_sec']:.0f}s")

    # сравнение
    h, b = report["human"], report["bot"]
    report["comparison"] = {
        "delta_n_clusters": b["n_clusters"] - h["n_clusters"],
        "delta_noise_fraction": b["noise_fraction"] - h["noise_fraction"],
        "delta_silhouette": b.get("silhouette", 0) - h.get("silhouette", 0),
    }
    plot_sizes(h, b)

    out = OUT / "wishart_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("\n=== ВЫВОД ===")
    print(f"  человек: {h['n_clusters']} кластеров, шум {h['noise_fraction']*100:.1f}%")
    print(f"  бот:     {b['n_clusters']} кластеров, шум {b['noise_fraction']*100:.1f}%")
    print(f"  Δ кластеров: {report['comparison']['delta_n_clusters']:+d}")


if __name__ == "__main__":
    main()
