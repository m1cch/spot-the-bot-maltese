"""
Шаг 7b — свип параметров Уишарта (k, h) по облакам человек/бот.

Цель: при h=0.2 в шум уходит ~78% точек — много. Прогоняем сетку (k, h),
ищем режим с умеренным шумом и максимальным расхождением human/bot
(по числу кластеров, доле шума, silhouette).

Запуск:
  python scripts/wishart_sweep.py --human datasets/human__n4_m8.npy \
         --bot datasets/bot__n4_m8.npy --ks 7,11,21,51 --hs 0.2,1,5,25

Выход:
  results/clustering/wishart_sweep.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wishart_compare import wishart, cluster_stats, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--bot", required=True)
    ap.add_argument("--ks", default="7,11,21,51")
    ap.add_argument("--hs", default="0.2,1,5,25")
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    hs = [float(x) for x in args.hs.split(",")]

    clouds = {}
    for name, path in (("human", args.human), ("bot", args.bot)):
        X = np.load(path).astype(np.float32)
        rng = np.random.default_rng(args.seed)
        if args.sample and len(X) > args.sample:
            X = X[rng.choice(len(X), args.sample, replace=False)]
        clouds[name] = X
        print(f"{name}: cloud {X.shape}")

    rows = []
    print(f"\n{'k':>4} {'h':>7} | {'H clu':>6} {'H noise':>8} {'H sil':>7} | "
          f"{'B clu':>6} {'B noise':>8} {'B sil':>7} | {'dClu':>5} {'dNoise':>7} {'dSil':>7}")
    print("-" * 88)
    for k in ks:
        for h in hs:
            st = {}
            for name in ("human", "bot"):
                lab = wishart(clouds[name], k, h)
                st[name] = cluster_stats(clouds[name], lab)
            H, B = st["human"], st["bot"]
            rows.append({"k": k, "h": h, "human": H, "bot": B})
            hs_, bs_ = H.get("silhouette", float("nan")), B.get("silhouette", float("nan"))
            print(f"{k:>4} {h:>7.2f} | {H['n_clusters']:>6} {H['noise_fraction']*100:>7.1f}% "
                  f"{hs_:>7.3f} | {B['n_clusters']:>6} {B['noise_fraction']*100:>7.1f}% "
                  f"{bs_:>7.3f} | {B['n_clusters']-H['n_clusters']:>5} "
                  f"{(B['noise_fraction']-H['noise_fraction'])*100:>6.1f}% "
                  f"{(B.get('silhouette',0)-H.get('silhouette',0)):>7.3f}", flush=True)

    out = OUT / "wishart_sweep.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
