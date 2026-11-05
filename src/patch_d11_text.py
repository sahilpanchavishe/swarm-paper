"""patch_d11_text.py -- bring paper/methods_draft.md and paper/results_discussion_draft.md in line with results/d11_paired.csv.
Run from the repo root:  python src/patch_d11_text.py
All-or-nothing: if any target sentence is not found exactly once, nothing is written."""
from pathlib import Path
import shutil, sys

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "paper" / "methods_draft.md"
R = ROOT / "paper" / "results_discussion_draft.md"

METHODS_STATS = (
    "Success rates are reported with 95 % Wilson confidence intervals. Comparisons are paired by (formation, seed, observability). "
    "Because success is a binary outcome, it is tested with the exact McNemar test, and the final error with a two-sided Wilcoxon "
    "signed-rank test. Each test is run separately for each training seed: a GNN trained with seed s is compared with the other GNN "
    "trained with seed s, and a comparison with a relay uses every training seed against the single relay run. A comparison is called "
    "significant only if it is significant in every training seed, so its p-value is the largest of the per-seed values. These p-values "
    "are corrected with the Holm method over the 16 comparisons of the pre-specified family. For each comparison we report the mean "
    "paired difference in success with a 95 % two-level bootstrap confidence interval (5000 resamples, in which both the test scenarios "
    "(formation, seed) and the training seeds are resampled) and the per-seed differences, because three training seeds make the seed "
    "level of the interval coarse. The primary hypothesis, fixed before any experiment, is that the K = 2 GNN has a higher success rate "
    "than the K = 1 GNN at 25 % observability. A result is called significant only if its Holm-corrected p is below 0.05 and the "
    "per-seed differences all point the same way; otherwise we say there is no evidence of a difference and report the width of the "
    "interval. The per-seed McNemar procedure replaced a first analysis that tested seed-averaged results with a Wilcoxon test; we made "
    "this change after the first analysis, because success is binary and averaging over training seeds hides their variation."
)

# (file, old, new) -- 'old' must occur exactly once
SUBS = [
    (R, "(95 % CI 20.8–29.7, Holm-corrected p < 0.0001)",
        "(95 % two-level bootstrap CI 18.2–32.7, Holm-corrected p = 2 × 10⁻⁹)"),
    (R, "The direction is the same for every training seed.",
        "The direction is the same for every training seed (per-seed differences 22.0, 31.0 and 22.5 points)."),
    (R, "CI 15.3–23.2, Holm p < 0.0001", "CI 13.0–26.3, Holm p < 0.0001"),
    (R, "(CI 12.7–19.5, Holm p < 0.0001)", "(CI 10.3–22.7; Holm p < 0.0001 for success, 0.039 for final error)"),
    (R, "(CI 7.2–17.3, Holm p < 0.0001 for success; p = 0.011 for final error)",
        "(CI 7.0–17.3, Holm p = 0.0004 for success and 0.0017 for final error)"),
    (R, "(CI 5.8–14.7, p = 0.0001 for success, but p = 0.20 for final error)",
        "(CI 5.8–14.8, Holm p = 0.0014 for success, but p = 1.0 for final error)"),
    (R, "At K = 1 the GNN is 33.3 points better.", "At K = 1 the GNN is 33.3 points better (CI 23.5–42.5)."),
    (R, "(+1.2 points, CI −1.3 to 4.0)", "(+1.2 points, CI −2.0 to 4.3)"),
    (R, "at K = 1 the GNN is slightly worse (−6.5 points, CI −12.8 to 0.0; Holm p = 0.22 for success and 0.027 for final error)",
        "at K = 1 the GNN is 6.5 points lower, a difference that is not significant (CI −14.5 to 1.5; Holm p = 1.0 for success and for final error)"),
    (R, "(Holm p = 0.031 for success, 0.44 for final error)",
        "(CI −0.2 to 3.5; Holm p = 1.0 for success and for final error)"),
]

texts = {M: M.read_text(encoding="utf-8"), R: R.read_text(encoding="utf-8")}
errors = []

# whole-line replacement of the Methods statistics paragraph
lines = texts[M].split("\n")
hit = [i for i, l in enumerate(lines) if l.startswith("Success rates are reported with 95 % Wilson")]
if len(hit) != 1:
    errors.append(f"methods statistics paragraph found {len(hit)} times (expected 1)")
else:
    lines[hit[0]] = METHODS_STATS
    texts[M] = "\n".join(lines)

for f, old, new in SUBS:
    n = texts[f].count(old)
    if n != 1:
        errors.append(f"{f.name}: found {n}x (expected 1): {old[:70]}...")
    else:
        texts[f] = texts[f].replace(old, new)

if errors:
    print("NOTHING WRITTEN. Problems:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

for f in (M, R):
    shutil.copy(f, f.with_suffix(".md.bak"))
    f.write_text(texts[f], encoding="utf-8")
print("patched", M.name, "and", R.name, "(backups: *.md.bak)")
print("Now check by hand: any other sentence about 'latched' and any remaining 'Wilcoxon', and the Table 2 numbers.")
