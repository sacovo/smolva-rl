#!/usr/bin/env python3
"""Harvest all eval results from outputs/eval/ into a per-model table.

Run on the cluster login node (cheap, file reads only):

    python3 scripts/cluster/harvest_eval.py [date-glob]

`date-glob` defaults to "*" (all dates); pass e.g. "2026-07-1*" to restrict.
Success rates come from eval_info.json -> overall.pc_success. Job dirs must
follow the lerobot-eval naming used by the rat_*.py scripts:
    <date>/<time>_recap_<model>_<suite>_cfg_<w>/eval_info.json
"""
import glob
import json
import os
import re
import sys

SUITES = ["libero_spatial", "libero_goal", "libero_10", "libero_object"]


def main():
    date_glob = sys.argv[1] if len(sys.argv) > 1 else "*"
    pattern = os.path.expanduser(
        f"~/smolvla-rl/outputs/eval/{date_glob}/*/eval_info.json"
    )
    res = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(
            r"recap_(\w+?)_(libero_(?:spatial|goal|10|object))_cfg_([\d.]+)/", p
        )
        if not m:
            continue
        model, suite, cfg = m.group(1), m.group(2), float(m.group(3))
        try:
            pc = json.load(open(p))["overall"]["pc_success"]
        except Exception:
            continue
        # Later runs of the same cell overwrite earlier ones (sorted glob).
        res.setdefault(model, {}).setdefault(suite, {})[cfg] = round(pc, 1)

    for model in sorted(res):
        print("==", model)
        for suite in SUITES:
            cells = res[model].get(suite, {})
            if not cells:
                continue
            best_w, best = max(cells.items(), key=lambda kv: kv[1])
            row = "  ".join(f"w{w}={v}" for w, v in sorted(cells.items()))
            print(f"  {suite}: {row}   BEST w{best_w}={best}")


if __name__ == "__main__":
    main()
