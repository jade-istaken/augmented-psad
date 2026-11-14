"""Paired bootstrap over test images for image-level AUROC (LA and SA).

Needs per-image scores saved by evaluate.py as <run_dir>/<category>/scores_k<k>.npz
(arrays scores_<type> and paths_<type> for the types good, logical_anomalies,
structural_anomalies).

Modes
-----
  --a RUN                 noise floor: AUROC with a 95% bootstrap CI per cell
  --a RUN --b RUN [RUN..] paired comparison: difference A - B with a 95% CI

Resampling is done within each category: good images are resampled once per
replicate and shared by LA and SA (they are the same images), while logical and
structural images are resampled separately. In compare mode, A and B are scored on
exactly the same resample, so the interval for A - B reflects only the part of the
noise that does not cancel. Category averages are the mean over categories of the
per-category AUROC on the same replicate.

This captures test-set sampling noise only. It does not cover the choice of
training images, random-init seeds, or the fact that configurations were selected
on the test set.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
TYPES = ["good", "logical_anomalies", "structural_anomalies"]


def auroc(neg, pos):
    """Rank-based AUROC; ties get average ranks. Higher score = more anomalous."""
    n_neg, n_pos = len(neg), len(pos)
    ranks = rankdata(np.concatenate([neg, pos]))
    return (ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_run(run_dir, category, k):
    path = Path(run_dir) / category / f"scores_k{k}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (was evaluate.py run with score saving?)")
    z = np.load(path)
    scores = {t: z[f"scores_{t}"].astype(np.float64) for t in TYPES}
    names = {t: [Path(str(p)).name for p in z[f"paths_{t}"]] for t in TYPES}
    return scores, names


def cell_aurocs(scores, idx):
    """(LA, SA) AUROC for one category on the resample idx = {type: indices}."""
    good = scores["good"][idx["good"]]
    la = auroc(good, scores["logical_anomalies"][idx["logical_anomalies"]])
    sa = auroc(good, scores["structural_anomalies"][idx["structural_anomalies"]])
    return la, sa


def bootstrap(runs, cats, n_boot, seed):
    """runs: list of {category: scores}. Returns point [R,C,2] and boot [R,B,C,2], in percent."""
    rng = np.random.default_rng(seed)
    n_runs = len(runs)
    point = np.zeros((n_runs, len(cats), 2))
    boot = np.zeros((n_runs, n_boot, len(cats), 2))

    for ci, c in enumerate(cats):
        full = {t: np.arange(len(runs[0][c][t])) for t in TYPES}
        for r in range(n_runs):
            point[r, ci] = cell_aurocs(runs[r][c], full)

    for b in range(n_boot):
        for ci, c in enumerate(cats):
            idx = {t: rng.integers(0, len(runs[0][c][t]), len(runs[0][c][t])) for t in TYPES}
            for r in range(n_runs):
                boot[r, b, ci] = cell_aurocs(runs[r][c], idx)
    return 100 * point, 100 * boot


def make_views(cats):
    """(label, function) pairs; each function maps [..., C, 2] arrays to a scalar per leading index."""
    views = []
    for i, c in enumerate(cats):
        views.append((f"{c} LA", lambda x, i=i: x[..., i, 0]))
        views.append((f"{c} SA", lambda x, i=i: x[..., i, 1]))
    views.append(("average LA", lambda x: x[..., 0].mean(axis=-1)))
    views.append(("average SA", lambda x: x[..., 1].mean(axis=-1)))
    views.append(("average overall", lambda x: x.mean(axis=(-2, -1))))
    return views


def check_against_json(run_dir, category, k, la, sa):
    """Warn if the recomputed AUROCs disagree with results_k<k>.json (stored as fractions)."""
    f = Path(run_dir) / category / f"results_k{k}.json"
    if not f.exists():
        return
    saved = json.loads(f.read_text())
    for key, val in (("LA_auroc", la), ("SA_auroc", sa)):
        if key in saved and abs(100 * saved[key] - val) > 1e-3:
            print(f"WARNING {run_dir}/{category}: recomputed {key} {val:.3f} "
                  f"differs from saved {100 * saved[key]:.3f}")


def load_all(run_dir, cats, k):
    scores, names = {}, {}
    for c in cats:
        scores[c], names[c] = load_run(run_dir, c, k)
    return scores, names


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="run directory (the save_dir holding <category>/ folders)")
    p.add_argument("--b", nargs="*", default=[], help="one or more run directories to compare against")
    p.add_argument("--categories", nargs="+", default=CATEGORIES)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="optional path to write the table")
    args = p.parse_args()

    cats = args.categories
    views = make_views(cats)
    scores_a, names_a = load_all(args.a, cats, args.k)

    print("Test images per category (good / logical / structural):")
    for c in cats:
        s = scores_a[c]
        print(f"  {c:<22}{len(s['good']):>5}{len(s['logical_anomalies']):>5}{len(s['structural_anomalies']):>5}")
    print()

    rows = []

    if not args.b:
        point, boot = bootstrap([scores_a], cats, args.n_boot, args.seed)
        for ci, c in enumerate(cats):
            check_against_json(args.a, c, args.k, *point[0, ci])
        print(f"{args.a}: AUROC (%) with 95% bootstrap CI, {args.n_boot} resamples")
        print(f"{'cell':<28}{'AUROC':>8}{'CI low':>9}{'CI high':>9}")
        for label, f in views:
            lo, hi = np.percentile(f(boot[0]), [2.5, 97.5])
            val = float(f(point[0]))
            print(f"{label:<28}{val:>8.2f}{lo:>9.2f}{hi:>9.2f}")
            rows.append({"run": args.a, "cell": label, "auroc": val, "ci_low": lo, "ci_high": hi})
    else:
        for b_dir in args.b:
            scores_b, names_b = load_all(b_dir, cats, args.k)
            for c in cats:
                for t in TYPES:
                    if names_a[c][t] != names_b[c][t]:
                        raise ValueError(f"{c}/{t}: A and B were evaluated on different images or in a "
                                         f"different order, so the pairing is invalid")
            point, boot = bootstrap([scores_a, scores_b], cats, args.n_boot, args.seed)
            for ci, c in enumerate(cats):
                check_against_json(args.a, c, args.k, *point[0, ci])
                check_against_json(b_dir, c, args.k, *point[1, ci])

            print(f"A = {args.a}\nB = {b_dir}\nDifference A - B in AUROC points, "
                  f"95% paired bootstrap CI ({args.n_boot} resamples); * = CI excludes 0")
            print(f"{'cell':<28}{'A':>8}{'B':>8}{'A-B':>8}{'CI low':>9}{'CI high':>9}")
            for label, f in views:
                a_val, b_val = float(f(point[0])), float(f(point[1]))
                diff = f(boot[0]) - f(boot[1])
                lo, hi = np.percentile(diff, [2.5, 97.5])
                flag = " *" if (lo > 0 or hi < 0) else ""
                print(f"{label:<28}{a_val:>8.2f}{b_val:>8.2f}{a_val - b_val:>8.2f}{lo:>9.2f}{hi:>9.2f}{flag}")
                rows.append({"a": args.a, "b": b_dir, "cell": label, "A": a_val, "B": b_val,
                             "diff": a_val - b_val, "ci_low": lo, "ci_high": hi})
            print()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved table to {args.csv}")


if __name__ == "__main__":
    main()