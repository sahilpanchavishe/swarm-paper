"""Day 15 (EXPLORATORY): classify the worst robot in every replayed failing episode. Run: python src/d15_classify.py"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "results" / "d15_replay_all.csv")
def cls(r):
    if r.worst_filter_active_pct > 50: return "held_by_filter"
    if r.worst_speed_mm_s >= 20: return "still_moving"
    if r.worst_nn_dist_mm > 100: return "stopped_isolated"
    return "stopped_in_crowd"
d["class"] = d.apply(cls, axis=1); d.to_csv(ROOT / "results" / "d15_failure_classes.csv", index=False)
print(f"{len(d)} failing episodes replayed, replay matches saved error in {int(d.match.sum())}")
print(d.groupby(["policy", "class"]).size().unstack(fill_value=0).to_string())
print(); print(d.groupby("class").agg(n=("class", "size"), median_init_comp=("init_comp", "median"), median_off_25mm=("robots_off_25mm", "median"),
                                      median_final_err_mm=("saved_err_mm", "median")).round(1).to_string())
print(); print("distinct (seed, formation, obs) starts:", d.groupby(["seed", "formation", "obs_pct"]).ngroups)
