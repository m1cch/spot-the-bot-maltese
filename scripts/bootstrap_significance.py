"""
Шаг 7d — оценка значимости различий человек/бот (Уишарт), корректная статистика.

Δ-метрики получены на одной подвыборке 30k 4-грамм. Здесь B раз берётся НОВАЯ
независимая подвыборка человека и бота, для каждой считается Уишарт. Получаем
для каждой метрики две независимые выборки по B значений (человек и бот).

Статистика (двухвыборочная — подвыборки независимы, пар между ними нет):
  - human / bot:  mean ± std распределения метрики;
  - Δ = mean(bot) − mean(human), стандартная ошибка SE = sqrt(var_h/B + var_b/B),
    95% доверительный интервал СРЕДНЕГО различия = Δ ± 1.96·SE;
  - критерий Манна–Уитни U (две независимые выборки) — значимо ли различие;
  - размер эффекта Коэна d = Δ / pooled_sd.

Запуск:
  python scripts/bootstrap_significance.py --human datasets/human__n4_m16.npy \
         --bot datasets/bot__n4_m16.npy -B 40

Выход:
  results/clustering/bootstrap_significance.json  (с сырыми B значениями)
"""
import argparse
import json
import math

import numpy as np
from scipy.stats import mannwhitneyu

from wishart_compare import wishart, cluster_stats, OUT

METRICS = ["n_clusters", "noise_fraction", "silhouette",
           "davies_bouldin", "calinski_harabasz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--bot", required=True)
    ap.add_argument("-k", "--neighbors", type=int, default=11)
    ap.add_argument("--hh", type=float, default=1.0)
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("-B", "--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Xh = np.load(args.human).astype(np.float32)
    Xb = np.load(args.bot).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    print(f"=== significance | B={args.iters} | k={args.neighbors} "
          f"h={args.hh} sample={args.sample} ===")
    print(f"human pool {Xh.shape} | bot pool {Xb.shape}\n")

    rec = {m: {"human": [], "bot": []} for m in METRICS}
    for it in range(args.iters):
        for name, X in (("human", Xh), ("bot", Xb)):
            sub = (X[rng.choice(len(X), args.sample, replace=False)]
                   if len(X) > args.sample else X)
            st = cluster_stats(sub, wishart(sub, args.neighbors, args.hh))
            for m in METRICS:
                rec[m][name].append(st.get(m, float("nan")))
        print(f"  iter {it+1}/{args.iters}", flush=True)

    report = {"params": {"B": args.iters, "k": args.neighbors, "h": args.hh,
                          "sample": args.sample}, "metrics": {}}
    print(f"\n{'metric':>20} | {'human (m±s)':>19} | {'bot (m±s)':>19} | "
          f"{'Δ mean':>9} {'95% CI Δ':>20} {'MW p':>10} {'Cohen d':>8}")
    print("-" * 116)
    for m in METRICS:
        h = np.array(rec[m]["human"], dtype=float)
        b = np.array(rec[m]["bot"], dtype=float)
        h, b = h[np.isfinite(h)], b[np.isfinite(b)]
        delta = b.mean() - h.mean()
        se = math.sqrt(h.var(ddof=1) / len(h) + b.var(ddof=1) / len(b))
        ci = (delta - 1.96 * se, delta + 1.96 * se)
        pooled = math.sqrt((h.var(ddof=1) + b.var(ddof=1)) / 2)
        cohen = delta / pooled if pooled > 0 else float("nan")
        try:
            _, p = mannwhitneyu(b, h, alternative="two-sided")
        except ValueError:
            p = float("nan")
        report["metrics"][m] = {
            "human_mean": float(h.mean()), "human_std": float(h.std(ddof=1)),
            "bot_mean": float(b.mean()), "bot_std": float(b.std(ddof=1)),
            "delta": float(delta), "se": float(se),
            "ci95": [float(ci[0]), float(ci[1])],
            "mannwhitney_p": float(p), "cohen_d": float(cohen),
            "raw_human": h.tolist(), "raw_bot": b.tolist(),
        }
        sig = "*" if (p < 0.05 and ci[0] * ci[1] > 0) else " "
        print(f"{m:>20} | {h.mean():>9.3f}±{h.std(ddof=1):<8.3f} | "
              f"{b.mean():>9.3f}±{b.std(ddof=1):<8.3f} | {delta:>9.3f} "
              f"[{ci[0]:>8.3f},{ci[1]:>8.3f}] {p:>10.2e} {cohen:>7.2f}{sig}")

    out = OUT / "bootstrap_significance.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    sig = [m for m in METRICS
           if report["metrics"][m]["mannwhitney_p"] < 0.05
           and report["metrics"][m]["ci95"][0] * report["metrics"][m]["ci95"][1] > 0]
    print(f"Значимо (p<0.05, CI не включает 0): {', '.join(sig) if sig else 'нет'}")


if __name__ == "__main__":
    main()
