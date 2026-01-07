# A-Deep-Learning-Based-Framework-for-Risk-Aware-Engineering-Training-in-Virtual-Reality


**1.Install RL libraries**
```
# 1) Install RL libraries (Kaggle usually allows pip)
!pip -q install stable-baselines3 gymnasium

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
```

**2.Load dataset**

```
# --------------------------------------------
# 2) Load dataset
# --------------------------------------------
CSV_PATH = "/kaggle/input/industrial-motor-data/industrial_motor_sensor_data_8000.csv"
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
```

**3.Auto-detect feature columns**

We try common names; if not found, we fallback to similar matches

```

# --------------------------------------------
# 3) Auto-detect feature columns (robust)
# --------------------------------------------
def find_col(possible_names, columns):
    cols_lower = {c.lower(): c for c in columns}
    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    # fallback: contains match
    for c in columns:
        cl = c.lower()
        for name in possible_names:
            if name.lower() in cl:
                return c
    return None

col_voltage = find_col(["Voltage", "voltage", "V"], df.columns)
col_current = find_col(["Current", "current", "I"], df.columns)
col_temp    = find_col(["Temperature", "temp", "temperature"], df.columns)
col_vib     = find_col(["Vibration", "vibration", "vib"], df.columns)

print("\nDetected columns:")
print("Voltage:", col_voltage)
print("Current:", col_current)
print("Temp:", col_temp)
print("Vibration:", col_vib)

# If any is missing, choose the first 4 numeric columns as fallback
needed = [col_voltage, col_current, col_temp, col_vib]
if any(c is None for c in needed):
    print("\n⚠️ Some expected columns not found. Falling back to first 4 numeric columns.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 4:
        raise ValueError("Not enough numeric columns in the dataset to build observations.")
    col_voltage, col_current, col_temp, col_vib = numeric_cols[:4]
    print("Fallback numeric columns:", [col_voltage, col_current, col_temp, col_vib])

features = [col_voltage, col_current, col_temp, col_vib]
data = df[features].copy()

# Handle NaNs just in case
data = data.replace([np.inf, -np.inf], np.nan).dropna()
data_arr = data.values.astype(np.float32)

print("\nFinal features used:", features)
print("Data array shape:", data_arr.shape)

```


**4.We normalize features (important for RL stability)**


```
eps = 1e-6
mean = data_arr.mean(axis=0, keepdims=True)
std  = data_arr.std(axis=0, keepdims=True) + eps
data_norm = (data_arr - mean) / std
```



**5.We define risk-aware reward helpers**


```
def compute_alarm_level(voltage_n, current_n, temp_n, vib_n):
    """
    Using normalized values (z-scores).
    We'll define alarm levels based on thresholds.
    """
    # These thresholds can be tuned.
    # z > 1.0 means above average; z > 2.0 means very high.
    alarm = 0
    if (temp_n > 1.0) or (vib_n > 1.0) or (current_n > 1.0):
        alarm = 1  # warning
    if (temp_n > 2.0) or (vib_n > 2.0) or (current_n > 2.0):
        alarm = 2  # critical
    return alarm

def compute_risk_score(voltage_n, current_n, temp_n, vib_n):
    """
    Risk score from normalized variables.
    You can adjust weights to match your scenario.
    """
    alarm = compute_alarm_level(voltage_n, current_n, temp_n, vib_n)
    risk = (
        0.2 * abs(voltage_n) +
        0.6 * max(0.0, current_n) +
        0.9 * max(0.0, temp_n) +
        1.0 * max(0.0, vib_n) +
        1.5 * alarm
    )
    return risk, alarm
```



**6.Build Gymnasium environment using real data**


```
class MotorRiskEnv(gym.Env):
    """
    Observation: [V, I, Temp, Vib, Alarm, Risk]  (all normalized except alarm is 0/1/2)
    Actions (Discrete):
      0 maintain
      1 reduce_load
      2 transfer_load
      3 call_maintenance
      4 isolate_section
      5 emergency_shutdown

    The dataset provides the evolving system state.
    Actions do NOT change the recorded sensor values (since they're real logs),
    but actions affect reward (penalty/bonus) based on risk and alarm levels.
    This is still a valid RL setup for learning a risk-aware policy over observed states.
    """
    metadata = {"render_modes": []}

    def __init__(self, data_norm, episode_len=200):
        super().__init__()
        self.data = data_norm
        self.N = len(self.data)
        self.episode_len = min(episode_len, self.N - 2)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )

        self.t = 0
        self.start_idx = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random starting point so agent sees diverse situations
        self.start_idx = self.np_random.integers(0, self.N - self.episode_len - 1)
        self.t = self.start_idx
        self.steps = 0

        v, i, temp, vib = self.data[self.t]
        risk, alarm = compute_risk_score(v, i, temp, vib)

        obs = np.array([v, i, temp, vib, float(alarm), float(risk)], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Get current state
        v, i, temp, vib = self.data[self.t]
        risk, alarm = compute_risk_score(v, i, temp, vib)

        # Move to next timestep from real log
        self.t += 1
        self.steps += 1

        # Next state (for obs)
        v2, i2, temp2, vib2 = self.data[self.t]
        risk2, alarm2 = compute_risk_score(v2, i2, temp2, vib2)
        obs = np.array([v2, i2, temp2, vib2, float(alarm2), float(risk2)], dtype=np.float32)

        # --------------------------
        # Reward shaping (Risk-aware)
        # --------------------------
        # Base: encourage low risk
        reward = 1.0 - 0.25 * risk

        # If critical alarm and you do nothing => penalize
        if alarm == 2 and action == 0:
            reward -= 3.0

        # If warning/critical and you take mitigating action => small bonus
        mitigating_actions = {1, 2, 3, 4}
        if alarm >= 1 and action in mitigating_actions:
            reward += 0.5

        # Emergency shutdown: strong mitigation but has cost
        if action == 5:
            # If risk is high, shutdown is justified
            if risk >= 4.5:
                reward += 1.0
            else:
                reward -= 1.0  # unnecessary shutdown cost

        # Termination on extreme risk (simulate failure)
        terminated = (risk >= 7.5) and (alarm == 2)
        truncated = (self.steps >= self.episode_len)

        info = {"risk": float(risk), "alarm": int(alarm), "action": int(action)}
        return obs, float(reward), terminated, truncated, info
```



**7.Train DQN**


```
env = MotorRiskEnv(data_norm=data_norm, episode_len=200)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    batch_size=64,
    learning_starts=2000,
    train_freq=4,
    target_update_interval=1000,
    gamma=0.99,
)

model.learn(total_timesteps=50000)
```



**8.Evaluate the trained policy**


```
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print("\nEvaluation mean reward:", mean_reward, "+/-", std_reward)
```

**9.Run one demo episode and print a small trace**


```
obs, _ = env.reset()
total_reward = 0.0

print("\n--- Demo Episode Trace (first 30 steps) ---")
for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    total_reward += reward

    if step < 30:
        print(f"step={step:02d} action={info['action']} alarm={info['alarm']} risk={info['risk']:.3f} reward={reward:.3f}")

    if terminated or truncated:
        break

print("\nTotal reward (demo episode):", total_reward)
```


**10.Save model**


```
model.save("motor_risk_dqn_model")
print("Saved model: motor_risk_dqn_model.zip")
```
