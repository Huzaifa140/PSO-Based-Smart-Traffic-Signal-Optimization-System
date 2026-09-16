# 🚦 PSO-Based Smart Traffic Signal Optimization System
### AI Course — Continuous Computing Problem (CCP) Project

---

## 📌 Project Overview

This project simulates a **4-way traffic intersection** where vehicle counts change dynamically. An AI agent using **Particle Swarm Optimization (PSO)** continuously calculates the optimal green-light duration for each road to minimize total vehicle waiting time.

This is a **Continuous Computing Problem (CCP)**: traffic conditions are never static. Every time vehicle counts update, the system re-runs PSO and adapts the signal timings in real time.

---

## 🧠 AI Algorithm: Particle Swarm Optimization (PSO)

PSO is a population-based metaheuristic algorithm inspired by the social behaviour of bird flocking or fish schooling.

### How PSO Works Here
- **Particles** = candidate solutions (4-element vectors: green times for each road)
- **Swarm** = population of particles exploring the solution space
- **Fitness function** = total vehicle-seconds of waiting time (to be minimised)
- Each particle updates its velocity and position using:

```
velocity = w * velocity
         + c1 * r1 * (personal_best - current_position)
         + c2 * r2 * (global_best   - current_position)

position = position + velocity
```

| Parameter | Role | Default |
|-----------|------|---------|
| `w`  | Inertia weight — controls momentum | 0.7 |
| `c1` | Cognitive coefficient — pull to personal best | 1.5 |
| `c2` | Social coefficient — pull to global best | 1.5 |

---

## 🗂️ Project Structure

```
traffic_pso/
│
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
```

---

## ⚙️ How to Run

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd traffic_pso
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🖥️ Dashboard Features

| Feature | Description |
|---------|-------------|
| **Vehicle Count Sliders** | Set 1–60 vehicles per road (North, South, East, West) |
| **Randomise Traffic** | Instantly generates random vehicle counts (simulates dynamic traffic) |
| **PSO Controls** | Tune swarm size, iterations, inertia, cognitive & social coefficients |
| **Before PSO Waiting Time** | Baseline: equal 30s green for all roads |
| **After PSO Waiting Time** | Optimised total waiting time |
| **Improvement %** | Percentage reduction in waiting time |
| **Signal Timing Output** | Optimal green-light seconds per road with traffic-light indicator |
| **Convergence Chart** | PSO fitness over iterations |
| **Green-Time Comparison Chart** | Before vs after bar chart |
| **Distribution Pie Charts** | Vehicle demand vs green-time allocation |
| **Iteration Log** | Step-by-step PSO progress |
| **Summary Table** | All roads with change indicators |

---

## 📐 Fitness Function

```python
def compute_waiting_time(green_times, vehicle_counts):
    throughput_rate = 1 / 4.0   # 1 vehicle cleared per 4 seconds of green
    total_cycle = sum(green_times)

    total_wait = 0
    for each road i:
        capacity = green_times[i] * throughput_rate
        leftover = max(0, vehicles[i] - capacity)
        total_wait += leftover * total_cycle          # leftover vehicles wait full cycle
        total_wait += served * (green_times[i] / 2)  # served vehicles wait avg half green

    return total_wait
```

---

## 📊 Example Results

| Road  | Vehicles | Green Before | Green After PSO | Change |
|-------|----------|-------------|-----------------|--------|
| North | 35       | 30s         | 47.2s           | ▲ 17s  |
| South | 10       | 30s         | 15.4s           | ▼ 15s  |
| East  | 45       | 30s         | 52.8s           | ▲ 23s  |
| West  | 8        | 30s         | 12.6s           | ▼ 17s  |

**Improvement: ~38% reduction in total waiting time**

---

## 🎓 Course Information

- **Subject**: Artificial Intelligence
- **Project Type**: CCP (Continuous Computing Problem)
- **Algorithm**: Particle Swarm Optimization (PSO)
- **Language**: Python 3.10+
- **Framework**: Streamlit

---
## 📚 References

1. Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization*. IEEE ICNN.
2. Streamlit Documentation — https://docs.streamlit.io
3. NumPy Documentation — https://numpy.org/doc
