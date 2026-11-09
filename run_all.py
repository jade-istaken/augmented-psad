import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
import gc
import torch

from build_memory_bank import build_bank
from evaluate import evaluate
from models import FeatureExtractor, init_backbone, feature_extractor

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]


def run_all(args):
    save_dir = Path(args.save_dir)
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(init_backbone(), layers_to_extract=args.layers).to(device).eval()

    for category in args.categories:
        print(f"\n===== {category} =====")
        bank_path = save_dir / category / "memory_bank.pt"

        if args.skip_build and bank_path.exists():
            print("Bank exists, skipping build")
        else:
            build_bank(Namespace(
                root_dir=args.root_dir,
                category=category,
                layers=args.layers,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                save_dir=args.save_dir,
                feature_extractor = extractor
            ))

        evaluate(Namespace(
            root_dir=args.root_dir,
            category=category,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_neighbors=args.num_neighbors,
            feature_extractor = extractor
        ))

        res = json.loads((save_dir / category / f"results_k{args.num_neighbors}.json").read_text())
        rows.append({"category": category, "LA_auroc": res["LA_auroc"], "SA_auroc": res["SA_auroc"]})
        gc.collect()
        torch.cuda.empty_cache()

    # averages over categories (AUROC reported as percentages to match the paper table)
    la = [r["LA_auroc"] for r in rows]
    sa = [r["SA_auroc"] for r in rows]
    rows.append({"category": "average", "LA_auroc": sum(la) / len(la), "SA_auroc": sum(sa) / len(sa)})
    for r in rows:
        r["LA_auroc"] = round(100 * r["LA_auroc"], 2)
        r["SA_auroc"] = round(100 * r["SA_auroc"], 2)

    out = save_dir / f"summary_k{args.num_neighbors}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "LA_auroc", "SA_auroc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'category':<22}{'LA':>8}{'SA':>8}")
    for r in rows:
        print(f"{r['category']:<22}{r['LA_auroc']:>8}{r['SA_auroc']:>8}")
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Build banks and evaluate all categories")
    p.add_argument("--root_dir", required=True)
    p.add_argument("--save_dir", default="./processed_memory_banks")
    p.add_argument("--categories", nargs="+", default=CATEGORIES)
    p.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    p.add_argument("--num_neighbors", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_build", action="store_true", help="Reuse existing banks")
    run_all(p.parse_args())