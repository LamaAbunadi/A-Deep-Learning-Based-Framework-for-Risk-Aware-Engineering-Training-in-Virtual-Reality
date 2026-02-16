# A-Deep-Learning-Based-Framework-for-Risk-Aware-Engineering-Training-in-Virtual-Reality

**Risk-Aware Deep Reinforcement Learning for Engineering Training**

Author: Lama A. Abunadi

**Overview:
    This code demonstrates a Deep Reinforcement Learning (DQN) framework
    for risk-aware decision training using industrial sensor data.**

Research Context:

    - Feasibility study for VR-based engineering training
    
    - Emphasis on decision sensitivity, not physical simulation
    

Disclaimer:
    This is not a physical simulator or digital twin.
    Transition dynamics are heuristic and safety-oriented.



```

# --------------------------------------------------
# Environment Design Rationale
# --------------------------------------------------
# Logged sensor data provides realistic baseline behavior.
# Action-dependent transitions are introduced to enable
# meaningful decision learning.
#
# This hybrid design is intentional and suitable for
# VR training feasibility studies.



# =========================
# Risk-Aware DQN Prototype (with action-responsive environment)
# =========================
# Idea:
# - We will use logged sensor data as a base signal (from CSV).
# - We add a simple rule-based transition model so actions
#   actually change the next state (directionally), making the RL
#   problem closer to a true MDP: (state, action) -> next_state.
```
```
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# =========================
# 1) Load CSV from Kaggle
# =========================
CSV_PATH = "/kaggle/input/industrial-motor-data/industrial_motor_sensor_data_8000.csv"
df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -------------------------
# Robust column detection
# -------------------------
def find_col(possible, columns):
    lower_map = {c.lower(): c for c in columns}
    for name in possible:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    # contains fallback
    for c in columns:
        cl = c.lower()
        for name in possible:
            if name.lower() in cl:
                return c
    return None

col_v = find_col(["voltage", "v"], df.columns)
col_i = find_col(["current", "i"], df.columns)
col_t = find_col(["temperature", "temp"], df.columns)
col_vib = find_col(["vibration", "vib"], df.columns)

needed = [col_v, col_i, col_t, col_vib]
if any(c is None for c in needed):
    print("⚠️ Some expected columns not found. Falling back to first 4 numeric columns.")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 4:
        raise ValueError("Not enough numeric columns found in the dataset.")
    col_v, col_i, col_t, col_vib = num_cols[:4]

features = [col_v, col_i, col_t, col_vib]
print("Using features:", features)

data = df[features].replace([np.inf, -np.inf], np.nan).dropna()
X = data.values.astype(np.float32)
print("Data used:", X.shape)

# =========================
# 2) Normalize for stability
# =========================
eps = 1e-6
mean = X.mean(axis=0, keepdims=True)
std  = X.std(axis=0, keepdims=True) + eps
Xn = (X - mean) / std

# =========================
# 3) Define Risk + Reward
# =========================
def alarm_level(v_n, i_n, t_n, vib_n):
    # thresholds on z-scores (tuneable)
    alarm = 0
    if (t_n > 1.0) or (vib_n > 1.0) or (i_n > 1.0):
        alarm = 1
    if (t_n > 2.0) or (vib_n > 2.0) or (i_n > 2.0):
        alarm = 2
    return alarm

def risk_score(v_n, i_n, t_n, vib_n):
    a = alarm_level(v_n, i_n, t_n, vib_n)
    risk = (
        0.2 * abs(v_n) +
        0.6 * max(0.0, i_n) +
        0.9 * max(0.0, t_n) +
        1.0 * max(0.0, vib_n) +
        1.5 * a
    )
    return risk, a

# Actions:
# 0 maintain
# 1 reduce_load
# 2 transfer_load
# 3 call_maintenance
# 4 isolate_section
# 5 emergency_shutdown
N_ACTIONS = 6

def compute_reward(state, action):
    v, i, t, vib = state
    risk, a = risk_score(v, i, t, vib)

    # base: prefer low risk
    reward = 1.0 - 0.25 * risk

    # If critical and do nothing => penalty
    if a == 2 and action == 0:
        reward -= 3.0

    # If warning/critical and take mitigating action => bonus
    mitigating = {1, 2, 3, 4}
    if a >= 1 and action in mitigating:
        reward += 0.5

    # shutdown: good if risk high, bad if unnecessary
    if action == 5:
        if risk >= 4.5:
            reward += 1.0
        else:
            reward -= 1.0

    # terminate condition (simulate failure)
    terminated = (risk >= 7.5 and a == 2)
    return float(reward), terminated, float(risk), int(a)


# --------------------------------------------------
# Action Semantics (Training-Oriented)
# --------------------------------------------------
# 0: Maintain normal operation
# 1: Reduce system load
# 2: Transfer load to alternative subsystem
# 3: Request maintenance intervention
# 4: Isolate affected system section
# 5: Emergency shutdown (high cost, high safety)


def apply_action_effect(state, action):
    """
    Apply a simple rule-based effect to the normalized state.
    state: np.array([v, i, t, vib]) normalized
    action: int in [0..5]
    returns: modified_state (np.float32)
    """
    v, i, t, vib = state.astype(np.float32).copy()

    if action == 0:
        # maintain: no intentional mitigation
        pass

    elif action == 1:  # reduce_load
        # Lower current, temperature gradually
        i *= 0.85
        t *= 0.90

    elif action == 2:  # transfer_load
        # Reduce current but may introduce slight voltage fluctuation
        i *= 0.80
        v += 0.05

    elif action == 3:  # call_maintenance
        # Maintenance tends to reduce vibration and temperature (after intervention)
        vib *= 0.70
        t *= 0.85

    elif action == 4:  # isolate_section
        # Strong reduction in current and vibration (isolation)
        i *= 0.50
        vib *= 0.50

    elif action == 5:  # emergency_shutdown
        # Drastic reduction in current and vibration; temperature drops fast
        i = 0.0
        vib = 0.0
        t *= 0.30

    # Optional: clamp values to avoid extreme explosion in prototype
    # (z-scores can drift if you keep adding offsets)
    v = float(np.clip(v, -5.0, 5.0))
    i = float(np.clip(i, -5.0, 5.0))
    t = float(np.clip(t, -5.0, 5.0))
    vib = float(np.clip(vib, -5.0, 5.0))

    return np.array([v, i, t, vib], dtype=np.float32)

# =========================
# 4) Environment over logged data (NOW action-responsive)
# =========================
class LoggedDataEnv:
    def __init__(self, Xn, episode_len=200, blend_alpha=0.70, noise_std=0.02):
        """
        Xn: normalized logged data
        episode_len: max steps per episode
        blend_alpha: how much we trust the logged base vs action-effect state
            next = alpha * base_logged + (1-alpha) * action_modified
        noise_std: small noise to avoid deterministic loops (prototype)
        """
        self.Xn = Xn
        self.N = len(Xn)
        self.episode_len = min(episode_len, self.N - 2)

        self.blend_alpha = float(blend_alpha)
        self.noise_std = float(noise_std)

        self.reset()

    def reset(self):
        self.start = np.random.randint(0, self.N - self.episode_len - 1)
        self.t = self.start
        self.steps = 0

        # Keep an internal "simulated" state that actions can affect.
        # Initialize it from the logged observation.
        self.sim_state = self._get_logged_obs()
        return self.sim_state.copy()

    def _get_logged_obs(self):
        # base observation from logged data at time t
        return self.Xn[self.t].astype(np.float32).copy()

    def _get_obs(self):
        # observation shown to the agent
        return self.sim_state.astype(np.float32).copy()

    def step(self, action):
        # Current simulated state
        state = self._get_obs()

        # Reward computed on current state (risk now)
        reward, terminated, risk, alarm = compute_reward(state, action)

        # Move time forward in the logged signal
        self.t += 1
        self.steps += 1

        # Base next state from logged data
        base_next = self._get_logged_obs()

        # Action-modified next state (from current state)
        action_next = apply_action_effect(state, action)

        # Blend them:
        # - If alpha is high, we follow logged trend more.
        # - If alpha is low, action has stronger control over next state.
        next_state = (
            self.blend_alpha * base_next +
            (1.0 - self.blend_alpha) * action_next
        )

        # Add small noise (optional)
        if self.noise_std > 0:
            next_state += np.random.normal(0.0, self.noise_std, size=next_state.shape).astype(np.float32)

        # Update internal sim state
        self.sim_state = next_state.astype(np.float32)

        done = terminated or (self.steps >= self.episode_len)
        info = {"risk": float(risk), "alarm": int(alarm)}
        return self._get_obs(), float(reward), bool(done), info

env = LoggedDataEnv(Xn, episode_len=200, blend_alpha=0.70, noise_std=0.02)

# =========================
# 5) DQN components
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class QNetwork(nn.Module):
    def __init__(self, state_dim=4, n_actions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

q = QNetwork(4, N_ACTIONS).to(device)
q_target = QNetwork(4, N_ACTIONS).to(device)
q_target.load_state_dict(q.state_dict())
q_target.eval()

optimizer = optim.Adam(q.parameters(), lr=1e-3)
loss_fn = nn.SmoothL1Loss()  # Huber

# Replay Buffer
buffer = deque(maxlen=50000)

def push(s, a, r, s2, d):
    buffer.append((s, a, r, s2, d))

def sample(batch_size=64):
    batch = random.sample(buffer, batch_size)
    s, a, r, s2, d = map(np.array, zip(*batch))
    return s, a, r, s2, d

# Epsilon-greedy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    with torch.no_grad():
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        qvals = q(st)
        return int(torch.argmax(qvals, dim=1).item())

# =========================
# 6) Train Loop
# =========================
gamma = 0.99
batch_size = 64
min_buffer = 2000
target_update_every = 500

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

num_episodes = 200
global_step = 0

for ep in range(1, num_episodes + 1):
    state = env.reset()
    ep_reward = 0.0
    ep_risk_sum = 0.0
    ep_alarm2 = 0

    done = False
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)

        push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        ep_risk_sum += info["risk"]
        if info["alarm"] == 2:
            ep_alarm2 += 1

        # learn
        if len(buffer) >= min_buffer:
            s, a, r, s2, d = sample(batch_size)

            s_t  = torch.tensor(s, dtype=torch.float32, device=device)
            a_t  = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
            r_t  = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
            s2_t = torch.tensor(s2, dtype=torch.float32, device=device)
            d_t  = torch.tensor(d.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

            # Q(s,a)
            q_sa = q(s_t).gather(1, a_t)

            # target: r + gamma * max_a' Q_target(s',a') * (1-done)
            with torch.no_grad():
                q_next_max = q_target(s2_t).max(dim=1, keepdim=True).values
                target = r_t + gamma * q_next_max * (1.0 - d_t)

            loss = loss_fn(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            optimizer.step()

            global_step += 1
            if global_step % target_update_every == 0:
                q_target.load_state_dict(q.state_dict())

    # epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    avg_risk = ep_risk_sum / max(1, env.steps)
    if ep % 10 == 0:
        print(
            f"Episode {ep:03d} | ep_reward={ep_reward:.2f} | "
            f"avg_risk={avg_risk:.2f} | critical_steps={ep_alarm2} | eps={epsilon:.2f}"
        )

print("Training done.")

# =========================
# 7) Quick Demo Run
# =========================
state = env.reset()
total_reward = 0.0
trace = []
for t in range(50):
    action = select_action(state, epsilon=0.0)  # greedy
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    trace.append((t, action, info["alarm"], info["risk"], reward))
    state = next_state
    if done:
        break

print("\nDemo (first 20 steps):")
for row in trace[:20]:
    t, a, al, rk, rw = row
    print(f"t={t:02d} action={a} alarm={al} risk={rk:.2f} reward={rw:.2f}")

print("Total demo reward:", total_reward)

```


**Save model**


```
model.save("motor_risk_dqn_model")
print("Saved model: motor_risk_dqn_model.zip")
```

**Example output**


```
Shape: (8000, 5)
Columns: ['Voltage (V)', 'Current (A)', 'Temperature (°C)', 'Vibration (mm/s)', 'Label']
   Voltage (V)  Current (A)  Temperature (°C)  Vibration (mm/s)     Label
0       460.01         1.79            118.33             22.37      high
1       419.12        15.52             36.93              2.18    normal
2       380.53        30.78             83.08             12.06  moderate
3       382.82        12.50             54.33              4.48    normal
4       333.79         0.96            118.99             24.41      high
Using features: ['Voltage (V)', 'Current (A)', 'Temperature (°C)', 'Vibration (mm/s)']
Data used: (8000, 4)
Device: cpu
Episode 010 | ep_reward=110.97 | avg_risk=1.19 | critical_steps=1 | eps=0.95
Episode 020 | ep_reward=128.48 | avg_risk=1.13 | critical_steps=1 | eps=0.90
Episode 030 | ep_reward=127.41 | avg_risk=1.28 | critical_steps=0 | eps=0.86
Episode 040 | ep_reward=137.78 | avg_risk=1.20 | critical_steps=0 | eps=0.82
Episode 050 | ep_reward=139.88 | avg_risk=1.29 | critical_steps=0 | eps=0.78
Episode 060 | ep_reward=139.52 | avg_risk=1.09 | critical_steps=0 | eps=0.74
Episode 070 | ep_reward=126.87 | avg_risk=1.26 | critical_steps=1 | eps=0.70
Episode 080 | ep_reward=135.08 | avg_risk=1.19 | critical_steps=0 | eps=0.67
Episode 090 | ep_reward=153.01 | avg_risk=0.99 | critical_steps=0 | eps=0.64
Episode 100 | ep_reward=142.07 | avg_risk=1.23 | critical_steps=0 | eps=0.61
Episode 110 | ep_reward=-0.58 | avg_risk=8.31 | critical_steps=1 | eps=0.58
Episode 120 | ep_reward=142.17 | avg_risk=1.08 | critical_steps=0 | eps=0.55
Episode 130 | ep_reward=-0.04 | avg_risk=8.17 | critical_steps=1 | eps=0.52
Episode 140 | ep_reward=145.08 | avg_risk=1.36 | critical_steps=0 | eps=0.50
Episode 150 | ep_reward=147.54 | avg_risk=1.36 | critical_steps=0 | eps=0.47
Episode 160 | ep_reward=147.94 | avg_risk=1.21 | critical_steps=0 | eps=0.45
Episode 170 | ep_reward=150.45 | avg_risk=1.22 | critical_steps=0 | eps=0.43
Episode 180 | ep_reward=152.11 | avg_risk=1.37 | critical_steps=2 | eps=0.41
Episode 190 | ep_reward=152.43 | avg_risk=1.24 | critical_steps=0 | eps=0.39
Episode 200 | ep_reward=158.70 | avg_risk=1.12 | critical_steps=1 | eps=0.37
Training done.

Demo (first 20 steps):
t=00 action=2 alarm=0 risk=0.11 reward=0.97
t=01 action=2 alarm=0 risk=1.27 reward=0.68
t=02 action=2 alarm=0 risk=0.20 reward=0.95
t=03 action=2 alarm=1 risk=4.36 reward=0.41
t=04 action=2 alarm=0 risk=0.57 reward=0.86
t=05 action=2 alarm=0 risk=0.03 reward=0.99
t=06 action=2 alarm=0 risk=0.05 reward=0.99
t=07 action=2 alarm=0 risk=1.68 reward=0.58
t=08 action=2 alarm=0 risk=1.61 reward=0.60
t=09 action=2 alarm=0 risk=0.17 reward=0.96
t=10 action=2 alarm=0 risk=0.10 reward=0.98
t=11 action=2 alarm=0 risk=0.99 reward=0.75
t=12 action=2 alarm=1 risk=3.75 reward=0.56
t=13 action=5 alarm=1 risk=5.26 reward=0.69
t=14 action=2 alarm=1 risk=3.38 reward=0.65
t=15 action=2 alarm=0 risk=1.68 reward=0.58
t=16 action=2 alarm=0 risk=0.97 reward=0.76
t=17 action=2 alarm=0 risk=0.41 reward=0.90
t=18 action=2 alarm=1 risk=3.90 reward=0.53
t=19 action=2 alarm=1 risk=4.25 reward=0.44
Total demo reward: 38.983556151390076
```

```
# --------------------------------------------------
# Limitations and Scope
# --------------------------------------------------
# - Transition dynamics are heuristic, not physics-based
# - Human factors (reaction delay, stress) are not modeled
#
# These choices are deliberate and align with early-stage
# doctoral feasibility exploration.
```
د
ذذذ
