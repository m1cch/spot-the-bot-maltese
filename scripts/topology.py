"""
Шаг 8 — топологический анализ облаков «человек» и «бот» (persistent homology).

По методичке Spot the bot: к n-граммным облакам применяется персистентная
гомология (Vietoris–Rips), сравниваются числа Бетти и диаграммы персистентности
человеческого и бот-корпуса.

Метрики (по Edelsbrunner & Harer, Zhu «Persistent Homology for NLP»):
  - числа Бетти b0, b1 (число значимых компонент / циклов);
  - суммарная персистентность по размерности;
  - макс./средняя персистентность;
  - энтропия персистентности  E = -Σ pᵢ·log pᵢ,  pᵢ = lifeᵢ / Σ life.

PH на облаке точек дорогая (VR-комплекс), поэтому облако подвыбирается
(--points, по умолчанию 2000) и усредняется по нескольким подвыборкам (--runs).

Вход:
  --human  datasets/human__n*_m*.npy
  --bot    datasets/bot__n*_m*.npy

Выход:
  results/topology/topology_report.json
  results/topology/persistence_diagrams.png

Запуск:
  python scripts/topology.py --human datasets/human__n4_m8.npy \
         --bot datasets/bot__n4_m8.npy --points 2000 --runs 5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "topology"
OUT.mkdir(parents=True, exist_ok=True)


def persistence(X, maxdim):
    """Диаграммы персистентности VR-комплекса. Возвращает list по размерностям."""
    from ripser import ripser
    res = ripser(X, maxdim=maxdim)
    return res["dgms"]


def diagram_stats(dgms, min_persistence):
    """Числа Бетти и статистики персистентности по каждой размерности."""
    stats = {}
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            stats[f"H{dim}"] = {"betti": 0, "total_persistence": 0.0,
                                "max_persistence": 0.0, "mean_persistence": 0.0,
                                "persistence_entropy": 0.0}
            continue
        life = dgm[:, 1] - dgm[:, 0]
        life = life[np.isfinite(life)]                 # отбросить бесконечную компоненту
        signif = life[life >= min_persistence]         # значимые особенности
        total = float(signif.sum())
        if total > 0:
            p = signif / total
            entropy = float(-np.sum(p * np.log(p + 1e-300)))
        else:
            entropy = 0.0
        stats[f"H{dim}"] = {
            "betti": int(len(signif)),
            "total_persistence": total,
            "max_persistence": float(signif.max()) if len(signif) else 0.0,
            "mean_persistence": float(signif.mean()) if len(signif) else 0.0,
            "persistence_entropy": entropy,
        }
    return stats


def analyse(X, points, runs, maxdim, min_persistence, seed):
    """Усреднить топологические метрики по нескольким подвыборкам облака."""
    rng = np.random.default_rng(seed)
    runs_stats, last_dgms = [], None
    for r in range(runs):
        sub = X if len(X) <= points else X[rng.choice(len(X), points, replace=False)]
        dgms = persistence(sub.astype(np.float32), maxdim)
        last_dgms = dgms
        runs_stats.append(diagram_stats(dgms, min_persistence))
    # усреднение по запускам
    avg = {}
    for dim_key in runs_stats[0]:
        avg[dim_key] = {}
        for metric in runs_stats[0][dim_key]:
            vals = [rs[dim_key][metric] for rs in runs_stats]
            avg[dim_key][metric] = {"mean": float(np.mean(vals)),
                                    "std": float(np.std(vals))}
    return avg, last_dgms


def plot_diagrams(dgms_h, dgms_b):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib нет — график пропущен)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dgms, name in ((axes[0], dgms_h, "human"), (axes[1], dgms_b, "bot")):
        lim = 0.0
        for dim, dgm in enumerate(dgms):
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite):
                lim = max(lim, finite.max())
                ax.scatter(finite[:, 0], finite[:, 1], s=10, label=f"H{dim}", alpha=0.6)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_title(f"{name}: диаграмма персистентности")
        ax.set_xlabel("рождение"); ax.set_ylabel("смерть")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "persistence_diagrams.png", dpi=120)
    print(f"  график: {OUT/'persistence_diagrams.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--bot", required=True)
    ap.add_argument("--points", type=int, default=2000, help="точек в подвыборке облака")
    ap.add_argument("--runs", type=int, default=5, help="число подвыборок для усреднения")
    ap.add_argument("--maxdim", type=int, default=1, help="макс. размерность гомологий")
    ap.add_argument("--min-persistence", type=float, default=0.0,
                    help="порог значимости особенности")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== Topology compare | points={args.points} runs={args.runs} "
          f"maxdim={args.maxdim} ===")
    report = {"params": vars(args)}
    dgms = {}
    for name, path in (("human", args.human), ("bot", args.bot)):
        X = np.load(path).astype(np.float32)
        print(f"\n[{name}] {path} -> cloud {X.shape}")
        t0 = time.time()
        avg, last = analyse(X, args.points, args.runs, args.maxdim,
                            args.min_persistence, args.seed)
        report[name] = avg
        dgms[name] = last
        for dk, st in avg.items():
            print(f"  {dk}: betti={st['betti']['mean']:.1f}±{st['betti']['std']:.1f} "
                  f"total_pers={st['total_persistence']['mean']:.3f} "
                  f"entropy={st['persistence_entropy']['mean']:.3f}")
        print(f"  {time.time()-t0:.0f}s")

    # сравнение
    cmp = {}
    for dk in report["human"]:
        cmp[dk] = {m: report["bot"][dk][m]["mean"] - report["human"][dk][m]["mean"]
                   for m in report["human"][dk]}
    report["comparison"] = cmp
    plot_diagrams(dgms["human"], dgms["bot"])

    out = OUT / "topology_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("\n=== ВЫВОД (Δ = бот − человек) ===")
    for dk, d in cmp.items():
        print(f"  {dk}: Δbetti={d['betti']:+.1f} "
              f"Δtotal_pers={d['total_persistence']:+.3f} "
              f"Δentropy={d['persistence_entropy']:+.3f}")


if __name__ == "__main__":
    main()
