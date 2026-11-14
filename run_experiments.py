"""Run every planned experiment, then collect the results into comparison tables.

Each experiment is one run_all.py call (build banks + evaluate all categories) in its own
subprocess, with its own save directory under --out_root. Subprocesses keep memory from
accumulating across runs and make every run independent.

Outputs (in --out_root):
  manifest.json      git commit, library versions, arguments, the planned experiments
  logs/              stdout and stderr of every run
  results_long.csv   one row per experiment x k x category (LA, SA)
  results_wide.csv   one row per experiment x k (averages + every category)
  results.md         grouped tables for reading and a ranking of all runs
  bootstrap/         paired-bootstrap tables (only with --bootstrap)

Typical use:
  python run_experiments.py --root_dir MVTec_LOCO_AD_512size
  python run_experiments.py --root_dir MVTec_LOCO_AD_512size --groups layers backbones --skip_existing
  python run_experiments.py --aggregate_only --bootstrap      # rebuild tables from existing runs
  python run_experiments.py --root_dir X --dry_run             # print the plan and commands only

The experiment plan is defined by the constants below; edit them to change what is run.
"""
import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
ABBR = {"breakfast_box": "BB", "juice_bottle": "JB", "pushpins": "PP",
        "screw_bag": "SB", "splicing_connectors": "SC"}
MODE_SHORT = {"joint": "joint", "layer_pixel": "pixel", "layer_pooled": "pooled"}

# ---------------------------------------------------------------- experiment plan
LAYER_ABLATION = ["1", "2", "3", "4", "12", "23", "34", "123"]       # CLIP RN101, pooled
NORM_ABLATION_LAYERS = ["23", "34", "123"]                            # joint / pixel / pooled
BACKBONES = ["clip_rn101", "clip_rn50", "resnet101_in1k_v2",          # at layers 1+2+3, pooled
             "wrn101_in1k_v2", "wrn50_in1k_v1", "dino_rn50"]
DEEP_BACKBONES = ["clip_rn101", "resnet101_in1k_v2"]                  # deeper-layer comparison
DEEP_LAYER_SETS = ["34", "4"]
RANDOM_BACKBONE = "random_clip_rn101"                                 # registry name prefix; seed appended
RANDOM_LAYER_SETS = ["123", "1"]
K_SWEEP = (1, 3, 5, 10)                                               # banks are reused across k
MAIN = "clip_rn101_l123_pooled"

GROUP_TITLES = OrderedDict([
    ("layers", "Layer ablation (CLIP RN101, pooled)"),
    ("norm_mode", "Normalization mode (CLIP RN101)"),
    ("backbones", "Backbones (layers 1+2+3, pooled)"),
    ("deep_layers", "CLIP vs ImageNet ResNet-101 at deeper layers (pooled)"),
    ("random_init", "Pretrained vs random initialization (pooled)"),
    ("k_sweep", "Number of neighbors k (CLIP RN101, layers 1+2+3, pooled)"),
])


@dataclass
class Experiment:
    backbone: str
    layers: str                      # digits of the stages used, e.g. "123"
    norm_mode: str = "layer_pooled"
    ks: tuple = (1,)
    groups: set = field(default_factory=set)

    @property
    def name(self):
        return f"{self.backbone}_l{self.layers}_{MODE_SHORT[self.norm_mode]}"

    @property
    def layer_arg(self):
        return "  ".join(f"layer{c}" for c in self.layers)


def build_plan(selected_groups, n_random_seeds):
    plan = OrderedDict()

    def add(group, backbone, layers, mode="layer_pooled", ks=(1,)):
        if group not in selected_groups:
            return
        exp = Experiment(backbone, layers, mode)
        cur = plan.setdefault(exp.name, exp)
        cur.groups.add(group)
        cur.ks = tuple(sorted(set(cur.ks) | set(ks)))

    for ls in LAYER_ABLATION:
        add("layers", "clip_rn101", ls)
    for ls in NORM_ABLATION_LAYERS:
        for mode in ("joint", "layer_pixel", "layer_pooled"):
            add("norm_mode", "clip_rn101", ls, mode)
    for bb in BACKBONES:
        add("backbones", bb, "123")
    for bb in DEEP_BACKBONES:
        for ls in DEEP_LAYER_SETS + ["123"]:
            add("deep_layers", bb, ls)
    for ls in RANDOM_LAYER_SETS:
        add("random_init", "clip_rn101", ls)
        for s in range(n_random_seeds):
            add("random_init", f"{RANDOM_BACKBONE}_s{s}", ls)
    add("k_sweep", "clip_rn101", "123", ks=K_SWEEP)
    return plan


def comparisons(n_random_seeds):
    """Paired-bootstrap comparisons, chosen in advance. Format: (label, A, [B, ...])."""
    return [
        ("noise_floor", MAIN, []),
        ("clip_vs_imagenet_l123", MAIN, ["resnet101_in1k_v2_l123_pooled"]),
        ("clip_vs_imagenet_l34", "clip_rn101_l34_pooled", ["resnet101_in1k_v2_l34_pooled"]),
        ("clip_vs_imagenet_l4", "clip_rn101_l4_pooled", ["resnet101_in1k_v2_l4_pooled"]),
        ("clip_vs_random", MAIN, [f"{RANDOM_BACKBONE}_s{s}_l123_pooled" for s in range(n_random_seeds)]),
        ("layers_123_vs_23", MAIN, ["clip_rn101_l23_pooled"]),
    ]


# ---------------------------------------------------------------- running
def find_summary(save_dir, k):
    d = Path(save_dir)
    hits = list(d.glob(f"summary_k{k}.csv")) + list(d.glob(f"summary_k{k}_*.csv"))
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": "unknown", "uncommitted_changes": None}


def library_versions():
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    return info


def run_experiment(exp, args, out_root, log_dir):
    save_dir = out_root / exp.name
    # a finished summary means the bank exists, so later k values can skip the build
    built = args.skip_existing and any(find_summary(save_dir, k) for k in exp.ks)
    for k in exp.ks:
        if args.skip_existing and find_summary(save_dir, k):
            print(f"  [skip] {exp.name} k={k} (summary exists)")
            continue
        cmd = [sys.executable, args.run_all_script,
               "--root_dir", args.root_dir, "--save_dir", str(save_dir),
               "--backbone", exp.backbone, "--layers", exp.layer_arg,
               "--norm_mode", exp.norm_mode, "--num_neighbors", str(k),
               "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
               "--seed", str(args.seed)]
        if built:
            cmd.append("--skip_build")
        if args.dry_run:
            print("  " + " ".join(cmd))
            built = True
            continue
        log = log_dir / f"{exp.name}_k{k}.log"
        start = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - start
        if rc != 0:
            print(f"  [FAILED] {exp.name} k={k} (exit code {rc}, see {log})")
            return False
        print(f"  [done] {exp.name} k={k} ({elapsed:.0f}s)")
        built = True
    return True


# ---------------------------------------------------------------- aggregation
def read_summary(path):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["category"]] = (float(row["LA_auroc"]), float(row["SA_auroc"]))
    return out


def collect(plan, out_root):
    records = []
    for exp in plan.values():
        for k in exp.ks:
            path = find_summary(out_root / exp.name, k)
            if path is None:
                continue
            s = read_summary(path)
            missing = [c for c in CATEGORIES if c not in s]
            if missing:
                print(f"warning: {path} lacks {missing}; skipping")
                continue
            la_avg, sa_avg = s["average"] if "average" in s else (
                statistics.mean(s[c][0] for c in CATEGORIES), statistics.mean(s[c][1] for c in CATEGORIES))
            rec = {"experiment": exp.name, "backbone": exp.backbone, "layers": exp.layers,
                   "norm_mode": exp.norm_mode, "k": k, "groups": sorted(exp.groups),
                   "kind": "seed" if re.search(r"_s\d+$", exp.backbone) else "run", "note": "",
                   "avg_LA": la_avg, "avg_SA": sa_avg, "avg_overall": (la_avg + sa_avg) / 2}
            for c in CATEGORIES:
                rec[f"{c}_LA"], rec[f"{c}_SA"] = s[c]
            records.append(rec)
    return records


def seed_means(records):
    """Mean over random-init seeds (and the min-max range of the averages) per configuration."""
    buckets = {}
    for r in records:
        if r["kind"] == "seed":
            base = re.sub(r"_s\d+$", "", r["backbone"])
            buckets.setdefault((base, r["layers"], r["norm_mode"], r["k"]), []).append(r)
    metrics = ["avg_LA", "avg_SA", "avg_overall"] + [f"{c}_{m}" for c in CATEGORIES for m in ("LA", "SA")]
    out = []
    for (base, layers, mode, k), rs in buckets.items():
        rec = {"experiment": f"{base}_l{layers}_{MODE_SHORT[mode]} (mean of {len(rs)} seeds)",
               "backbone": base, "layers": layers, "norm_mode": mode, "k": k,
               "groups": sorted(set(g for r in rs for g in r["groups"])), "kind": "mean"}
        for m in metrics:
            rec[m] = statistics.mean(r[m] for r in rs)
        rec["note"] = "range of seed averages: " + "; ".join(
            f"{lab} {min(r[m] for r in rs):.1f}-{max(r[m] for r in rs):.1f}"
            for lab, m in (("LA", "avg_LA"), ("SA", "avg_SA"), ("overall", "avg_overall")))
        out.append(rec)
    return out


def write_csvs(records, out_root):
    wide_cols = ["experiment", "groups", "backbone", "layers", "norm_mode", "k", "kind", "note",
                 "avg_LA", "avg_SA", "avg_overall"] + [f"{c}_{m}" for c in CATEGORIES for m in ("LA", "SA")]
    with open(out_root / "results_wide.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=wide_cols)
        writer.writeheader()
        for r in records:
            row = {c: r.get(c, "") for c in wide_cols}
            row["groups"] = ";".join(r["groups"])
            for c in wide_cols[8:]:
                row[c] = f"{r[c]:.2f}"
            writer.writerow(row)

    long_cols = ["experiment", "backbone", "layers", "norm_mode", "k", "kind", "category", "LA", "SA"]
    with open(out_root / "results_long.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=long_cols)
        writer.writeheader()
        for r in records:
            for c in CATEGORIES + ["average"]:
                la, sa = (r["avg_LA"], r["avg_SA"]) if c == "average" else (r[f"{c}_LA"], r[f"{c}_SA"])
                writer.writerow({"experiment": r["experiment"], "backbone": r["backbone"],
                                 "layers": r["layers"], "norm_mode": r["norm_mode"], "k": r["k"],
                                 "kind": r["kind"], "category": c, "LA": f"{la:.2f}", "SA": f"{sa:.2f}"})


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def table_rows(recs):
    rows = []
    for r in recs:
        cells = [r["experiment"], str(r["k"]), f"{r['avg_LA']:.1f}", f"{r['avg_SA']:.1f}",
                 f"{r['avg_overall']:.1f}"]
        cells += [f"{r[f'{c}_LA']:.1f}" for c in CATEGORIES] + [f"{r[f'{c}_SA']:.1f}" for c in CATEGORIES]
        rows.append(cells)
    return rows


def write_markdown(records, out_root, meta):
    headers = (["experiment", "k", "LA avg", "SA avg", "overall"] +
               [f"LA {ABBR[c]}" for c in CATEGORIES] + [f"SA {ABBR[c]}" for c in CATEGORIES])
    lines = ["# Results", "",
             f"Generated {meta['generated']}. Image-level AUROC (%). LA = logical, SA = structural; "
             "BB = breakfast_box, JB = juice_bottle, PP = pushpins, SB = screw_bag, "
             "SC = splicing_connectors.",
             f"Commit {meta['git']['commit'][:10]}"
             f"{' (uncommitted changes)' if meta['git']['uncommitted_changes'] else ''}; "
             f"batch size {meta['batch_size']}.", ""]
    for group, title in GROUP_TITLES.items():
        recs = [r for r in records if group in r["groups"]]
        if not recs:
            continue
        lines += [f"## {title}", ""] + md_table(headers, table_rows(recs)) + [""]
        for r in recs:
            if r["note"]:
                lines.append(f"- {r['experiment']}: {r['note']}")
        if any(r["note"] for r in recs):
            lines.append("")
    ranked = sorted((r for r in records if r["kind"] != "seed" and r["k"] == 1),
                    key=lambda r: -r["avg_overall"])
    seen, unique = set(), []
    for r in ranked:
        if r["experiment"] not in seen:
            seen.add(r["experiment"])
            unique.append(r)
    lines += ["## All runs ranked by overall average (k = 1)", ""]
    lines += md_table(["rank", "experiment", "LA avg", "SA avg", "overall"],
                      [[str(i + 1), r["experiment"], f"{r['avg_LA']:.1f}", f"{r['avg_SA']:.1f}",
                        f"{r['avg_overall']:.1f}"] for i, r in enumerate(unique)])
    lines += ["", "Rankings reflect test-set scores. Differences smaller than the test-sampling noise "
              "(about 2 points on the overall average) should not be read as real; see bootstrap/.", ""]
    (out_root / "results.md").write_text("\n".join(lines))


# ---------------------------------------------------------------- bootstrap
def run_bootstraps(args, out_root):
    boot_dir = out_root / "bootstrap"
    boot_dir.mkdir(exist_ok=True)
    for label, a, bs in comparisons(args.random_seeds):
        runs = [a] + bs
        missing = [r for r in runs if not (out_root / r / CATEGORIES[0] / "scores_k1.npz").exists()]
        if missing:
            print(f"  [skip] bootstrap {label}: no per-image scores for {missing}")
            continue
        cmd = [sys.executable, args.bootstrap_script, "--a", str(out_root / a), "--k", "1",
               "--n_boot", str(args.n_boot), "--csv", str(boot_dir / f"{label}.csv")]
        if bs:
            cmd += ["--b"] + [str(out_root / b) for b in bs]
        res = subprocess.run(cmd, capture_output=True, text=True)
        (boot_dir / f"{label}.txt").write_text(res.stdout + res.stderr)
        print(f"  [{'done' if res.returncode == 0 else 'FAILED'}] bootstrap {label}")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root_dir", help="path to the MVTec LOCO dataset (not needed with --aggregate_only)")
    p.add_argument("--out_root", default="runs/final")
    p.add_argument("--groups", nargs="+", default=list(GROUP_TITLES), choices=list(GROUP_TITLES))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random_seeds", type=int, default=3,
                   help="random-init seeds; the registry must define that many (it loops over range(3) by default)")
    p.add_argument("--skip_existing", action="store_true", help="skip runs whose summary already exists")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--aggregate_only", action="store_true", help="only rebuild the tables")
    p.add_argument("--bootstrap", action="store_true", help="run the planned paired-bootstrap comparisons")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--run_all_script", default="run_all.py")
    p.add_argument("--bootstrap_script", default="bootstrap_compare.py")
    args = p.parse_args()

    if not args.aggregate_only and not args.root_dir:
        p.error("--root_dir is required unless --aggregate_only is set")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(set(args.groups), args.random_seeds)
    n_runs = sum(len(e.ks) for e in plan.values())
    meta = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "git": git_info(),
            "versions": library_versions(), "batch_size": args.batch_size, "args": vars(args),
            "experiments": [{"name": e.name, "backbone": e.backbone, "layers": e.layer_arg,
                             "norm_mode": e.norm_mode, "ks": list(e.ks), "groups": sorted(e.groups)}
                            for e in plan.values()]}

    if not args.aggregate_only:
        print(f"{len(plan)} experiments, {n_runs} evaluations planned")
        if not args.dry_run:
            (out_root / "manifest.json").write_text(json.dumps(meta, indent=2, default=str))
        log_dir = out_root / "logs"
        log_dir.mkdir(exist_ok=True)
        failures = []
        for i, exp in enumerate(plan.values(), start=1):
            print(f"[{i}/{len(plan)}] {exp.name}  (k = {', '.join(map(str, exp.ks))})")
            if not run_experiment(exp, args, out_root, log_dir):
                failures.append(exp.name)
                if args.stop_on_error:
                    break
        if args.dry_run:
            return
        if failures:
            print("\nFailed experiments:\n  " + "\n  ".join(failures))

    records = collect(plan, out_root)
    records += seed_means(records)
    if not records:
        print("No results found to aggregate.")
        return
    write_csvs(records, out_root)
    write_markdown(records, out_root, meta)
    print(f"\nWrote {out_root / 'results.md'}, results_wide.csv and results_long.csv")

    if args.bootstrap:
        run_bootstraps(args, out_root)

    if not args.aggregate_only and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()