"""Accuracy of stage-3 move counting against known labels.

    python tools/evaluate_counts.py                       # your practice runs (practice_data/)
    python tools/evaluate_counts.py --set on=0.2 rot_hi=0.65   # re-analyse them with other thresholds
    python tools/evaluate_counts.py --synthetic 12        # rendered clips (vision/synth.py)
    python tools/evaluate_counts.py --stored              # use the results saved at record time

Practice runs store compact per-frame motion numbers (no images), so the analysis
is re-run here: change a threshold with --set and see the effect on all your
recordings at once. Reports, per run and in total: measured vs true turns,
rotations, unclassified, half turns, and for paced runs whether the deliberate
pauses were found and how well turning/paused periods agree with the cues
(after estimating your reaction lag).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from vision.motion_analysis import AnalysisParams  # noqa: E402
from vision.motion_eval import Report, evaluate_practice, evaluate_synthetic_clip, run_synthetic  # noqa: E402
from vision.practice import DEFAULT_SEQUENCES, PracticeStore  # noqa: E402


def parse_sets(items: list[str], base: dict) -> AnalysisParams:
    d = dict(base)
    for it in items:
        k, _, v = it.partition("=")
        if k not in AnalysisParams.__dataclass_fields__:
            raise SystemExit(f"unknown parameter {k!r}; choose from {', '.join(AnalysisParams.__dataclass_fields__)}")
        d[k] = v
    return AnalysisParams.from_dict(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--data", default=None, help="practice data folder (default: practice.path from the config)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="analysis parameter overrides")
    ap.add_argument("--stored", action="store_true", help="score the results saved at record time (no re-analysis)")
    ap.add_argument("--synthetic", type=int, default=0, metavar="N", help="evaluate N rendered clips instead")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--last", type=int, default=0, help="only the last N practice runs")
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.config)
    params = parse_sets(args.set, cfg["motion"].get("analysis") or {})
    changed = {k: v for k, v in asdict(params).items() if v != getattr(AnalysisParams(), k)}
    print("analysis parameters:", changed or "defaults")

    if args.synthetic:
        rep = Report()
        for i in range(args.synthetic):
            spec = DEFAULT_SEQUENCES[i % len(DEFAULT_SEQUENCES)]
            clip = run_synthetic(spec, args.seed + i)
            row, _ = evaluate_synthetic_clip(clip, params)
            rep.add(row)
            print(rep.format().splitlines()[-2], flush=True)
        print()
        print(rep.format(f"Synthetic clips ({args.synthetic})"))
        return 0

    root = Path(args.data) if args.data else ROOT / cfg["practice"]["path"]
    runs = PracticeStore(root).load()
    if args.last:
        runs = runs[-args.last:]
    if not runs:
        print(f"No practice runs in {root}. Record some with P (practice mode) in the app.")
        return 1
    rep = evaluate_practice(root, runs, params, reanalyse=not args.stored)
    print(rep.format(f"Practice runs in {root}"))
    by_spec: dict[str, Report] = {}
    for row in rep.rows:
        by_spec.setdefault(row.name.split(" ", 1)[-1], Report()).add(row)
    if len(by_spec) > 1:
        print("\nPer sequence:")
        for spec, r in by_spec.items():
            t = r.totals()
            print(f"  {spec[:50]:50s} runs {t['runs']:2d}  turns err {t['turn_err_pct']:5.1f}% (bias {t['turn_bias_pct']:+5.1f}%)"
                  f"  rot |err| {t['rot_abs_err']}/{t['true_rotations']}  unc {t['unclassified']}"
                  f"  half |err| {t['half_abs_err']}/{t['true_halves']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
