"""Validation of the MuJoCo e-puck model against ideal unicycle kinematics. Run: python test_mj_env.py"""
import numpy as np, time
from mj_env import *
env = SwarmEnv()
p0 = np.array([[(i%4)*0.4-0.6, (i//4)*0.6-0.3] for i in range(N)]); th0 = np.zeros(N)

def run(v, w, T):
    env.reset(p0, th0); steps = int(T/DT_CTRL); vv = np.full(N, v); ww = np.full(N, w)
    for _ in range(steps): p, th = env.step(vv, ww)
    return p, th, steps*DT_CTRL

print("--- straight, v=0.10 m/s, 3 s ---")
p, th, T = run(0.10, 0.0, 3.0); d = np.linalg.norm(p-p0, axis=1)
print(f"ideal {0.10*T:.3f} m | measured mean {d.mean():.3f} m (min {d.min():.3f}, max {d.max():.3f})")
print("--- spin in place, w=2 rad/s, 2 s ---")
p, th, T = run(0.0, 2.0, 2.0); dth = np.abs(np.unwrap(th)-0)
print(f"ideal yaw {2*T:.2f} rad (wrapped {np.arctan2(np.sin(2*T),np.cos(2*T)):.2f}) | measured yaw {np.round(th[:3],2)} | drift {np.linalg.norm(p-p0,axis=1).max():.3f} m")
print("--- arc, v=0.08, w=0.5, 4 s ---")
p, th, T = run(0.08, 0.5, 4.0)
R = 0.08/0.5; ix = p0[:,0]+R*np.sin(0.5*T); iy = p0[:,1]+R*(1-np.cos(0.5*T))
print(f"ideal end offset from ideal pos: {np.linalg.norm(p-np.c_[ix,iy],axis=1).mean():.4f} m (mean over robots)")
print("--- speed ---")
env.reset(p0, th0); t=time.time()
for _ in range(550): env.step(np.full(N,0.05), np.zeros(N))
print(f"one 550-step (17.6 s) episode of 8 robots: {time.time()-t:.2f} s wall-clock")
