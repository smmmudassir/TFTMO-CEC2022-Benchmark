#!/usr/bin/env python3
"""Turn validated statistical CSVs into conservative manuscript-ready result sentences."""
from pathlib import Path
import argparse, pandas as pd, json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stats",default="final/stats")
    ap.add_argument("--validation",default="final/validation_status.json")
    ap.add_argument("--out",default="final/RESULTS_SUMMARY.md")
    a=ap.parse_args()
    v=json.loads(Path(a.validation).read_text())
    if not (v.get("full_complete") and v.get("ablation_complete")):
        raise SystemExit("Refusing to write claims from incomplete evidence.")
    s=Path(a.stats)
    ranks=pd.read_csv(s/"full_average_ranks.csv")
    wtl=pd.read_csv(s/"win_tie_loss.csv")
    tests=pd.read_csv(s/"wilcoxon_holm.csv")
    lines=["# Frozen CTC-DE evidence summary","",
           "All statements below are generated from the validated frozen campaign; they do not alter the algorithm.","",
           "## Average ranks","",ranks.to_markdown(index=False),"","## Win / tie / loss","",wtl.to_markdown(index=False),"",
           "## Holm-corrected pairwise tests",""]
    for comp,g in tests.groupby("comparator"):
        sig=g[g.reject_holm.astype(bool)]
        better=sig[sig.median_ctc < sig.median_other]
        worse=sig[sig.median_ctc > sig.median_other]
        lines.append(f"- **{comp}:** {len(better)} significant CTC-DE advantages, {len(worse)} significant disadvantages, {len(g)-len(sig)} non-significant problem-wise comparisons after Holm correction.")
    Path(a.out).write_text("\n".join(lines)+"\n")
    print(a.out)

if __name__=="__main__": main()
