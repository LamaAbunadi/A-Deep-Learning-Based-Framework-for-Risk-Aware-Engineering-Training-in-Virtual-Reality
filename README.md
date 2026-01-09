# A-Deep-Learning-Based-Framework-for-Risk-Aware-Engineering-Training-in-Virtual-Reality


**1.Install RL libraries**
```
# 1) Install RL libraries (Kaggle usually allows pip)
!pip -q install stable-baselines3 gymnasium


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

# =========================
# 4) "Environment" over logged data
# =========================
class LoggedDataEnv:
    def __init__(self, Xn, episode_len=200):
        self.Xn = Xn
        self.N = len(Xn)
        self.episode_len = min(episode_len, self.N - 2)
        self.reset()

    def reset(self):
        self.start = np.random.randint(0, self.N - self.episode_len - 1)
        self.t = self.start
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # observation = 4 features only (you can append alarm/risk if you want)
        return self.Xn[self.t].copy()

    def step(self, action):
        state = self._get_obs()
        reward, terminated, risk, alarm = compute_reward(state, action)

        self.t += 1
        self.steps += 1
        next_state = self._get_obs()

        done = terminated or (self.steps >= self.episode_len)
        info = {"risk": risk, "alarm": alarm}
        return next_state, reward, done, info

env = LoggedDataEnv(Xn, episode_len=200)

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

num_episodes = 200  # increase if you want
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
        print(f"Episode {ep:03d} | ep_reward={ep_reward:.2f} | avg_risk={avg_risk:.2f} | critical_steps={ep_alarm2} | eps={epsilon:.2f}")

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
