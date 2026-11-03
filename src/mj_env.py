"""
mj_env.py -- MuJoCo e-puck swarm environment (headless, planar, differential drive with real wheel contact).
The e-puck has NO official MuJoCo model, so we build one from the published e-puck dimensions.
Validate it (see test_mj_env.py) and describe the validation in the paper's Methods.
"""
import numpy as np, mujoco

N = 8
R_WHEEL, AXLE, ROBOT_R = 0.0205, 0.052, 0.037
W_WHEEL_MAX = 6.28                      # rad/s, wheel limit
V_MAX = W_WHEEL_MAX * R_WHEEL           # 0.1287 m/s
ARENA = 1.1
TAU_MAX = 0.004                          # N*m per wheel (e-puck class stepper; tunable, document in Methods)
DT_CTRL = 0.032                         # control step (s)
DT_PHYS = 0.004                         # MuJoCo step (s)
SUBSTEPS = int(round(DT_CTRL/DT_PHYS))  # 8

def build_xml(n=N):
    bodies, acts = [], []
    for i in range(n):
        bodies.append(f"""
    <body name="r{i}" pos="{i*0.2-0.7} 0 {R_WHEEL}">
      <freejoint name="fj{i}"/>
      <geom name="chassis{i}" type="cylinder" size="{ROBOT_R} 0.0265" pos="0 0 0.008" mass="0.15" contype="2" conaffinity="2" friction="0.3 0.005 0.0001" rgba="0.2 0.5 0.9 1"/>
      <geom name="front{i}" type="box" size="0.004 0.012 0.004" pos="{ROBOT_R-0.004} 0 0.036" mass="0.0001" contype="0" conaffinity="0" rgba="1 0 0 1"/>
      <geom name="skidF{i}" type="sphere" size="0.005" pos="0.03 0 {-R_WHEEL+0.006}" mass="0.005" condim="1" friction="0 0 0" contype="1" conaffinity="0"/>
      <geom name="skidB{i}" type="sphere" size="0.005" pos="-0.03 0 {-R_WHEEL+0.006}" mass="0.005" condim="1" friction="0 0 0" contype="1" conaffinity="0"/>
      <body name="wl{i}" pos="0 {AXLE/2} 0">
        <joint name="jl{i}" type="hinge" axis="0 1 0" damping="1e-5"/>
        <geom type="cylinder" size="{R_WHEEL} 0.002" euler="90 0 0" mass="0.01" friction="1.5 0.005 0.0001" contype="1" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wr{i}" pos="0 {-AXLE/2} 0">
        <joint name="jr{i}" type="hinge" axis="0 1 0" damping="1e-5"/>
        <geom type="cylinder" size="{R_WHEEL} 0.002" euler="90 0 0" mass="0.01" friction="1.5 0.005 0.0001" contype="1" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>""")
        acts.append(f'<velocity name="al{i}" joint="jl{i}" kv="0.05" forcerange="-{TAU_MAX} {TAU_MAX}" ctrlrange="{-W_WHEEL_MAX} {W_WHEEL_MAX}"/>')
        acts.append(f'<velocity name="ar{i}" joint="jr{i}" kv="0.05" forcerange="-{TAU_MAX} {TAU_MAX}" ctrlrange="{-W_WHEEL_MAX} {W_WHEEL_MAX}"/>')
    a, h = ARENA, 0.05
    return f"""<mujoco model="epuck_swarm">
  <option timestep="{DT_PHYS}" gravity="0 0 -9.81" integrator="implicit"/>
  <default><geom solref="0.01 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="{a+0.1} {a+0.1} 0.1" friction="1 0.005 0.0001"/>
    <geom type="box" contype="2" conaffinity="2" pos="{a+0.01} 0 {h/2}" size="0.01 {a+0.02} {h/2}"/>
    <geom type="box" contype="2" conaffinity="2" pos="{-a-0.01} 0 {h/2}" size="0.01 {a+0.02} {h/2}"/>
    <geom type="box" contype="2" conaffinity="2" pos="0 {a+0.01} {h/2}" size="{a+0.02} 0.01 {h/2}"/>
    <geom type="box" contype="2" conaffinity="2" pos="0 {-a-0.01} {h/2}" size="{a+0.02} 0.01 {h/2}"/>
    {''.join(bodies)}
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""

class SwarmEnv:
    def __init__(self, n=N):
        self.n = n
        self.model = mujoco.MjModel.from_xml_string(build_xml(n))
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(f"fj{i}").qposadr[0] for i in range(n)]
        self.act_l = [self.model.actuator(f"al{i}").id for i in range(n)]
        self.act_r = [self.model.actuator(f"ar{i}").id for i in range(n)]

    def reset(self, p, theta):
        mujoco.mj_resetData(self.model, self.data)
        for i, a in enumerate(self.qadr):
            self.data.qpos[a:a+3] = [p[i, 0], p[i, 1], R_WHEEL]
            self.data.qpos[a+3:a+7] = [np.cos(theta[i]/2), 0, 0, np.sin(theta[i]/2)]   # yaw quaternion (w,x,y,z)
        mujoco.mj_forward(self.model, self.data)

    def poses(self):
        p = np.zeros((self.n, 2)); th = np.zeros(self.n)
        for i, a in enumerate(self.qadr):
            x, y = self.data.qpos[a], self.data.qpos[a+1]
            w, qx, qy, qz = self.data.qpos[a+3:a+7]
            p[i] = [x, y]; th[i] = np.arctan2(2*(w*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        return p, th

    def step(self, v, w):
        """Apply unicycle command (v m/s, w rad/s) for one control step via wheel speeds."""
        wl = (v - w*AXLE/2)/R_WHEEL; wr = (v + w*AXLE/2)/R_WHEEL
        for i in range(self.n):
            self.data.ctrl[self.act_l[i]] = wl[i]; self.data.ctrl[self.act_r[i]] = wr[i]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        return self.poses()
