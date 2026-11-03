"""Day 2a: validate the MuJoCo e-puck against ideal unicycle kinematics.
Run: python validate_model.py
Writes: ../results/model_validation.csv and ../figures/model_validation.png (for the paper's Methods section)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mj_env import SwarmEnv, N, DT_CTRL, AXLE, R_WHEEL

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "results").mkdir(exist_ok=True); (ROOT / "figures").mkdir(exist_ok=True)
env = SwarmEnv()
p0 = np.array([[(i % 4) * 0.4 - 0.6, (i // 4) * 0.6 - 0.3] for i in range(N)]); th0 = np.zeros(N)

def run(v, w, T):
    """Apply constant (v, w); return time, positions (steps,2) and unwrapped yaw of robot 0."""
    env.reset(p0, th0); steps = int(round(T / DT_CTRL)); vv = np.full(N, v); ww = np.full(N, w)
    P, TH = [p0[0].copy()], [0.0]
    for _ in range(steps):
        p, th = env.step(vv, ww); P.append(p[0].copy()); TH.append(th[0])
    return np.arange(steps + 1) * DT_CTRL, np.array(P), np.unwrap(np.array(TH))

def ideal(v, w, t):
    th = w * t
    if abs(w) < 1e-9: return np.c_[p0[0, 0] + v * t, np.full_like(t, p0[0, 1])], th
    R = v / w
    return np.c_[p0[0, 0] + R * np.sin(th), p0[0, 1] + R * (1 - np.cos(th))], th

cases = {"straight (v=0.10 m/s, 3 s)": (0.10, 0.0, 3.0),
         "spin (w=2 rad/s, 2 s)": (0.0, 2.0, 2.0),
         "arc (v=0.08 m/s, w=0.5 rad/s, 4 s)": (0.08, 0.5, 4.0)}
rows, data = [], {}
for name, (v, w, T) in cases.items():
    t, P, th = run(v, w, T); Pi, thi = ideal(v, w, t); data[name] = (t, P, th, Pi, thi)
    path_ideal = np.sum(np.linalg.norm(np.diff(Pi, axis=0), axis=1))
    path_meas = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    rows.append(dict(case=name, ideal_path_m=path_ideal, measured_path_m=path_meas,
                     path_error_pct=100 * (path_meas - path_ideal) / path_ideal if path_ideal > 0 else np.nan,
                     end_position_error_m=np.linalg.norm(P[-1] - Pi[-1]),
                     ideal_yaw_rad=thi[-1], measured_yaw_rad=th[-1],
                     yaw_error_pct=100 * (th[-1] - thi[-1]) / thi[-1] if abs(thi[-1]) > 0 else np.nan))
df = pd.DataFrame(rows); df.to_csv(ROOT / "results" / "model_validation.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(df.round(4).to_string(index=False))

# time constant of the speed response (straight case): time to reach 63% of commanded speed
t, P, th, Pi, thi = data["straight (v=0.10 m/s, 3 s)"]
speed = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1) / DT_CTRL]
tau = t[np.argmax(speed >= 0.632 * 0.10)]
print(f"speed response: 63% of commanded speed after {tau:.3f} s")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
t, P, th, Pi, thi = data["arc (v=0.08 m/s, w=0.5 rad/s, 4 s)"]
ax[0].plot(Pi[:, 0], Pi[:, 1], "k--", label="ideal unicycle"); ax[0].plot(P[:, 0], P[:, 1], label="MuJoCo e-puck")
ax[0].set_aspect("equal"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].set_title("Arc path"); ax[0].legend()
t, P, th, Pi, thi = data["straight (v=0.10 m/s, 3 s)"]
ax[1].plot(t, speed, label="MuJoCo e-puck"); ax[1].axhline(0.10, color="k", ls="--", label="commanded")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("speed (m/s)"); ax[1].set_title("Speed step response"); ax[1].legend()
t, P, th, Pi, thi = data["spin (w=2 rad/s, 2 s)"]
ax[2].plot(t, thi, "k--", label="ideal"); ax[2].plot(t, th, label="MuJoCo e-puck")
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("yaw (rad)"); ax[2].set_title("Spin in place"); ax[2].legend()
fig.tight_layout(); fig.savefig(ROOT / "figures" / "model_validation.png", dpi=300)
print("saved results/model_validation.csv and figures/model_validation.png")
