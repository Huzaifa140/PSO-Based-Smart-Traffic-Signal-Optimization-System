"""
PSO-Based Smart Traffic Signal Optimization System
===================================================
CCP Project | AI Course | Python + Streamlit
Author: Student Project
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import random

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic Signal Optimizer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark traffic-control aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-card .label {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
}
.metric-card .unit {
    font-size: 13px;
    color: #8b949e;
    margin-left: 4px;
}

/* Road badges */
.road-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    margin-right: 6px;
}

/* Signal lights */
.signal-block {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.signal-green  { color: #3fb950; }
.signal-yellow { color: #d29922; }
.signal-red    { color: #f85149; }

/* Section headers */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58a6ff;
    margin: 28px 0 12px 0;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #238636, #3fb950);
}

/* Improve buttons */
.stButton > button {
    background: #238636;
    color: #ffffff;
    border: 1px solid #2ea043;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 22px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #2ea043;
    border-color: #3fb950;
}

/* Sliders */
.stSlider > label { color: #8b949e !important; font-size: 13px; }

/* Info box */
.info-box {
    background: #0d2137;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 14px;
    line-height: 1.6;
    color: #c9d1d9;
}

/* PSO iteration log */
.pso-log {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #3fb950;
    max-height: 200px;
    overflow-y: auto;
}

/* improvement banner */
.improve-banner {
    background: linear-gradient(135deg, #0d2137 0%, #0d1f0f 100%);
    border: 1px solid #3fb950;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.improve-pct {
    font-family: 'Share Tech Mono', monospace;
    font-size: 48px;
    color: #3fb950;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PSO CORE ALGORITHM
# ─────────────────────────────────────────────

def compute_waiting_time(green_times: np.ndarray, vehicle_counts: np.ndarray) -> float:
    """
    Improved Fitness function for PSO.
    Penalizes leftover vehicles exponentially to force the AI to clear heavy traffic.
    """
    throughput_rate = 1.0  # 1 car crosses per second
    lost_time = 12.0  # Yellow/Red transition time
    total_cycle = np.sum(green_times) + lost_time

    total_wait = 0.0
    for i in range(4):
        capacity = green_times[i] * throughput_rate
        leftover = max(0, vehicle_counts[i] - capacity)

        # 🚨 THE FIX: Heavy exponential penalty for cars stuck waiting multiple cycles
        queue_penalty = (leftover ** 2) * 10.0

        # Normal wait time for cars that get through the green light
        served = min(vehicle_counts[i], capacity)
        red_time = total_cycle - green_times[i]
        normal_delay = served * (red_time / 2.0)

        total_wait += queue_penalty + normal_delay

    return total_wait


def run_pso(vehicle_counts: np.ndarray,
            n_particles: int = 40,
            n_iterations: int = 80,
            w: float = 0.7,
            c1: float = 1.5,
            c2: float = 1.5,
            min_green: float = 10.0,
            max_green: float = 60.0) -> dict:
    """
    Particle Swarm Optimization to find optimal green-light timings.

    Each particle is a 4-element vector: [g1, g2, g3, g4] seconds.
    The swarm minimises compute_waiting_time().

    Parameters
    ----------
    vehicle_counts : current vehicles per road
    n_particles    : swarm population size
    n_iterations   : how many rounds the swarm runs
    w              : inertia weight (momentum)
    c1             : cognitive coefficient (pull toward personal best)
    c2             : social coefficient   (pull toward global best)
    min_green/max_green : bounds for each green-time variable

    Returns
    -------
    dict with best positions, best fitness, history, and logs
    """
    dim = 4  # one dimension per road

    # ── Initialise positions & velocities ──────────────────────────────
    positions  = np.random.uniform(min_green, max_green, (n_particles, dim))
    velocities = np.random.uniform(-5, 5, (n_particles, dim))

    personal_best_pos = positions.copy()
    personal_best_val = np.array([compute_waiting_time(p, vehicle_counts) for p in positions])

    global_best_idx = np.argmin(personal_best_val)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_val = personal_best_val[global_best_idx]

    fitness_history = []   # track convergence
    pso_log         = []   # human-readable iteration log

    # ── Main PSO loop ───────────────────────────────────────────────────
    for iteration in range(n_iterations):
        r1 = np.random.rand(n_particles, dim)  # random weights – cognitive
        r2 = np.random.rand(n_particles, dim)  # random weights – social

        # Update velocity: inertia + cognitive pull + social pull
        velocities = (w * velocities
                      + c1 * r1 * (personal_best_pos - positions)
                      + c2 * r2 * (global_best_pos   - positions))

        # Clamp velocity to prevent explosion
        velocities = np.clip(velocities, -10, 10)

        # Update positions and enforce bounds
        positions = positions + velocities
        positions = np.clip(positions, min_green, max_green)

        # Evaluate fitness for each particle
        for p_idx in range(n_particles):
            fitness = compute_waiting_time(positions[p_idx], vehicle_counts)

            # Update personal best
            if fitness < personal_best_val[p_idx]:
                personal_best_val[p_idx] = fitness
                personal_best_pos[p_idx] = positions[p_idx].copy()

            # Update global best
            if fitness < global_best_val:
                global_best_val = fitness
                global_best_pos = positions[p_idx].copy()

        fitness_history.append(global_best_val)

        # Log every 10 iterations
        if (iteration + 1) % 10 == 0 or iteration == 0:
            pso_log.append(
                f"[Iter {iteration+1:3d}]  Best fitness = {global_best_val:8.2f} "
                f"| Timings = {np.round(global_best_pos, 1)}"
            )

    return {
        "best_green_times" : global_best_pos,
        "best_fitness"     : global_best_val,
        "fitness_history"  : fitness_history,
        "pso_log"          : pso_log,
    }


def naive_waiting_time(vehicle_counts: np.ndarray, equal_green: float = 30.0) -> float:
    """Baseline: equal green time for all roads (no optimisation)."""
    green_times = np.full(4, equal_green)
    return compute_waiting_time(green_times, vehicle_counts)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

ROAD_NAMES   = ["North", "South", "East", "West"]
ROAD_COLORS  = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]
ROAD_EMOJIS  = ["⬆️", "⬇️", "➡️", "⬅️"]


def signal_emoji(green_time: float) -> str:
    if green_time >= 40:
        return "🟢"
    elif green_time >= 20:
        return "🟡"
    else:
        return "🔴"


# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────

if "vehicle_counts" not in st.session_state:
    st.session_state.vehicle_counts = [20, 15, 25, 10]
if "pso_result" not in st.session_state:
    st.session_state.pso_result = None
if "auto_running" not in st.session_state:
    st.session_state.auto_running = False
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0


# ─────────────────────────────────────────────
# SIDEBAR – CONTROLS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚦 Control Panel")

    st.markdown('<p class="section-header">Vehicle Count Per Road</p>', unsafe_allow_html=True)

    counts = []
    for i, name in enumerate(ROAD_NAMES):
        val = st.slider(
            f"{ROAD_EMOJIS[i]} {name} Road",
            min_value=1, max_value=60,
            value=st.session_state.vehicle_counts[i],
            key=f"slider_{i}"
        )
        counts.append(val)
    st.session_state.vehicle_counts = counts

    st.markdown("---")
    st.markdown('<p class="section-header">PSO Parameters</p>', unsafe_allow_html=True)

    n_particles  = st.slider("Swarm size (particles)", 10, 100, 40, step=5)
    n_iterations = st.slider("Iterations",             20, 200, 80, step=10)
    inertia_w    = st.slider("Inertia weight (w)",    0.3, 1.0, 0.7, step=0.05)
    c1           = st.slider("Cognitive coeff (c1)",  0.5, 2.5, 1.5, step=0.1)
    c2           = st.slider("Social coeff (c2)",     0.5, 2.5, 1.5, step=0.1)

    st.markdown("---")
    st.markdown('<p class="section-header">Simulation Mode</p>', unsafe_allow_html=True)

    if st.button("🎲 Randomise Traffic", use_container_width=True):
        st.session_state.vehicle_counts = [random.randint(1, 60) for _ in range(4)]
        st.session_state.pso_result = None
        st.rerun()

    st.caption("Simulates real-world dynamic traffic changes.")


# ─────────────────────────────────────────────
# MAIN PAGE – HEADER
# ─────────────────────────────────────────────

st.markdown("""
<h1 style='font-family:"Share Tech Mono",monospace; font-size:28px; color:#58a6ff; margin-bottom:4px;'>
🚦 PSO-Based Smart Traffic Signal Optimizer
</h1>
<p style='color:#8b949e; font-size:14px; margin-top:0;'>
AI Course CCP Project &nbsp;|&nbsp; Particle Swarm Optimization &nbsp;|&nbsp; Python + Streamlit
</p>
<hr style='border-color:#21262d; margin:16px 0;'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ABOUT / EXPLANATION
# ─────────────────────────────────────────────

with st.expander("📖 How This System Works", expanded=False):
    st.markdown("""
    <div class="info-box">
    <b>What is this?</b><br>
    This system simulates a 4-way traffic intersection. Vehicle counts on each road change
    dynamically (like real traffic). The AI uses <b>Particle Swarm Optimization (PSO)</b>
    to calculate the optimal green-light duration for each road to minimise total waiting time.
    <br><br>
    <b>What is PSO?</b><br>
    PSO is a nature-inspired AI algorithm based on the flocking behaviour of birds.
    A "swarm" of particles explores the solution space simultaneously. Each particle:
    <br>• Remembers its own best position (cognitive memory)
    <br>• Moves toward the swarm's best-known position (social learning)
    <br>• Balances exploration and exploitation via inertia weight
    <br><br>
    <b>Continuous Computing Problem:</b><br>
    Traffic is never static. Every time vehicles are updated (manually or randomly),
    the system re-runs PSO and finds new optimal timings — adapting in real time.
    <br><br>
    <b>Fitness Function:</b>
    Total vehicle-seconds of waiting, accounting for throughput capacity per green window.
    PSO minimises this value.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CURRENT TRAFFIC STATUS
# ─────────────────────────────────────────────

st.markdown('<p class="section-header">Current Traffic Status</p>', unsafe_allow_html=True)

vehicle_counts = np.array(st.session_state.vehicle_counts, dtype=float)
total_vehicles = int(vehicle_counts.sum())
max_road       = ROAD_NAMES[int(np.argmax(vehicle_counts))]

col1, col2, col3, col4, col5 = st.columns(5)
cols = [col1, col2, col3, col4]
for i, col in enumerate(cols):
    with col:
        bar_pct = int((vehicle_counts[i] / 60) * 100)
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{ROAD_EMOJIS[i]} {ROAD_NAMES[i]}</div>
          <div class="value" style="color:{ROAD_COLORS[i]}">{int(vehicle_counts[i])}</div>
          <div class="unit">vehicles</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(bar_pct)

with col5:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Total Queue</div>
      <div class="value">{total_vehicles}</div>
      <div class="unit">vehicles</div>
    </div>
    <div class="metric-card" style="margin-top:6px;">
      <div class="label">Busiest Road</div>
      <div class="value" style="font-size:18px;">{max_road}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PRE-PSO BASELINE
# ─────────────────────────────────────────────

st.markdown('<p class="section-header">Baseline — Equal Green Time (No Optimization)</p>', unsafe_allow_html=True)

baseline_wait = naive_waiting_time(vehicle_counts, equal_green=30.0)
equal_green_times = np.full(4, 30.0)

bc1, bc2 = st.columns([3, 1])
with bc1:
    st.markdown("""
    <div class="info-box">
    Without optimization, each road gets a fixed <b>30-second</b> green light regardless
    of how many vehicles are waiting. This is the current state of most traditional traffic signals.
    </div>
    """, unsafe_allow_html=True)
with bc2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">⏱ Baseline Waiting Time</div>
      <div class="value" style="color:#f85149;">{baseline_wait:,.0f}</div>
      <div class="unit">veh·sec</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PSO OPTIMISE BUTTON
# ─────────────────────────────────────────────

st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run_pso_btn = st.button("⚡ Run PSO Optimization", use_container_width=True)

if run_pso_btn:
    progress_bar = st.progress(0)
    status_text  = st.empty()

    # Animate progress while PSO runs
    status_text.markdown("🔄 Initialising swarm...")
    progress_bar.progress(5)
    time.sleep(0.2)

    status_text.markdown("🔄 Running PSO iterations...")
    progress_bar.progress(20)

    result = run_pso(
        vehicle_counts,
        n_particles=n_particles,
        n_iterations=n_iterations,
        w=inertia_w,
        c1=c1,
        c2=c2,
    )

    progress_bar.progress(90)
    status_text.markdown("✅ Convergence achieved — finalising...")
    time.sleep(0.3)
    progress_bar.progress(100)
    time.sleep(0.2)
    progress_bar.empty()
    status_text.empty()

    st.session_state.pso_result = result
    st.session_state.run_counter += 1


# ─────────────────────────────────────────────
# PSO RESULTS
# ─────────────────────────────────────────────

if st.session_state.pso_result:
    res         = st.session_state.pso_result
    best_times  = res["best_green_times"]
    best_wait   = res["best_fitness"]
    improvement = ((baseline_wait - best_wait) / baseline_wait) * 100 if baseline_wait > 0 else 0

    st.markdown('<p class="section-header">PSO Optimization Results</p>', unsafe_allow_html=True)

    # ── Improvement Banner ──────────────────────────────────────────────
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">⏱ Before PSO</div>
          <div class="value" style="color:#f85149;">{baseline_wait:,.0f}</div>
          <div class="unit">veh·sec</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">⚡ After PSO</div>
          <div class="value" style="color:#3fb950;">{best_wait:,.0f}</div>
          <div class="unit">veh·sec</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
        <div class="improve-banner">
          <div class="label" style="color:#8b949e; font-size:11px; letter-spacing:0.1em; text-transform:uppercase;">
            🎯 Improvement
          </div>
          <div class="improve-pct">{improvement:.1f}%</div>
          <div style="color:#8b949e; font-size:12px;">reduction in waiting time</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Optimised Signal Timings ────────────────────────────────────────
    st.markdown('<p class="section-header">Optimized Green-Light Timings</p>', unsafe_allow_html=True)

    tc1, tc2, tc3, tc4 = st.columns(4)
    timing_cols = [tc1, tc2, tc3, tc4]
    for i, col in enumerate(timing_cols):
        with col:
            g = best_times[i]
            s_emoji = signal_emoji(g)
            st.markdown(f"""
            <div class="signal-block">
              <div style="font-size:32px;">{s_emoji}</div>
              <div style="font-family:'Share Tech Mono',monospace; font-size:22px;
                          color:{ROAD_COLORS[i]}; font-weight:700;">
                {g:.1f}s
              </div>
              <div style="font-size:12px; color:#8b949e; margin-top:4px;">
                {ROAD_EMOJIS[i]} {ROAD_NAMES[i]}
              </div>
              <div style="font-size:11px; color:#8b949e;">
                🚗 {int(vehicle_counts[i])} vehicles
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Visual Analysis</p>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Chart 1: Green time comparison (before vs after)
    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor("#161b22")
        ax1.set_facecolor("#0d1117")

        x = np.arange(4)
        bar_w = 0.35
        bars_b = ax1.bar(x - bar_w/2, equal_green_times, bar_w,
                         label="Before PSO (equal)", color="#f85149", alpha=0.85)
        bars_a = ax1.bar(x + bar_w/2, best_times, bar_w,
                         label="After PSO (optimal)", color="#3fb950", alpha=0.85)

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{e}\n{n}" for e, n in zip(ROAD_EMOJIS, ROAD_NAMES)],
                            color="#c9d1d9", fontsize=10)
        ax1.set_ylabel("Green Time (seconds)", color="#8b949e", fontsize=10)
        ax1.set_title("Green-Light Duration: Before vs After PSO",
                      color="#e6edf3", fontsize=11, pad=12)
        ax1.tick_params(colors="#8b949e")
        ax1.spines[:].set_color("#21262d")
        ax1.legend(facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=9)

        for bar in bars_a:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:.0f}s", ha="center", color="#3fb950", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # Chart 2: PSO convergence curve
    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor("#161b22")
        ax2.set_facecolor("#0d1117")

        history = res["fitness_history"]
        ax2.plot(history, color="#58a6ff", linewidth=2, label="Best Fitness")
        ax2.axhline(y=baseline_wait, color="#f85149", linestyle="--",
                    linewidth=1.2, label=f"Baseline = {baseline_wait:,.0f}")
        ax2.axhline(y=best_wait, color="#3fb950", linestyle="--",
                    linewidth=1.2, label=f"PSO Best = {best_wait:,.0f}")

        ax2.fill_between(range(len(history)), history, best_wait,
                         alpha=0.15, color="#58a6ff")

        ax2.set_xlabel("Iteration", color="#8b949e", fontsize=10)
        ax2.set_ylabel("Total Waiting Time (veh·sec)", color="#8b949e", fontsize=10)
        ax2.set_title("PSO Convergence Curve", color="#e6edf3", fontsize=11, pad=12)
        ax2.tick_params(colors="#8b949e")
        ax2.spines[:].set_color("#21262d")
        ax2.legend(facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Chart 3: Vehicle count vs allocated green time (pie-like bar)
    st.markdown('<p class="section-header">Vehicle Demand vs Green-Time Allocation</p>',
                unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 3.5))
    fig3.patch.set_facecolor("#161b22")

    for ax in ax3:
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#21262d")
        ax.tick_params(colors="#8b949e")

    # Pie – vehicle distribution
    wedge_props = dict(width=0.4, edgecolor="#0d1117", linewidth=2)
    ax3[0].pie(vehicle_counts, labels=ROAD_NAMES, colors=ROAD_COLORS,
               autopct="%1.0f%%", pctdistance=0.75,
               textprops={"color": "#e6edf3", "fontsize": 10},
               wedgeprops=wedge_props)
    ax3[0].set_title("Vehicle Distribution", color="#e6edf3", fontsize=11)

    # Donut – green time allocation
    ax3[1].pie(best_times, labels=ROAD_NAMES, colors=ROAD_COLORS,
               autopct="%1.0f%%", pctdistance=0.75,
               textprops={"color": "#e6edf3", "fontsize": 10},
               wedgeprops=wedge_props)
    ax3[1].set_title("Green-Time Allocation (PSO)", color="#e6edf3", fontsize=11)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── PSO Iteration Log ────────────────────────────────────────────────
    with st.expander("🔍 PSO Iteration Log", expanded=False):
        log_html = "<br>".join(res["pso_log"])
        st.markdown(f'<div class="pso-log">{log_html}</div>', unsafe_allow_html=True)
        st.caption(f"Run #{st.session_state.run_counter} | "
                   f"{n_particles} particles × {n_iterations} iterations")

    # ── Summary Table ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Summary Table</p>', unsafe_allow_html=True)
    import pandas as pd

    df = pd.DataFrame({
        "Road"              : [f"{ROAD_EMOJIS[i]} {ROAD_NAMES[i]}" for i in range(4)],
        "Vehicles"          : [int(v) for v in vehicle_counts],
        "Green (Before) s"  : [30] * 4,
        "Green (After PSO) s": [round(t, 1) for t in best_times],
        "Change"            : [f"{'▲' if best_times[i]>30 else '▼'} {abs(best_times[i]-30):.1f}s"
                               for i in range(4)],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    # Prompt to run PSO
    st.markdown("""
    <div class="info-box" style="text-align:center; padding: 30px;">
      <div style="font-size:40px;">⚡</div>
      <div style="font-size:16px; color:#58a6ff; font-weight:600; margin-top:10px;">
        Set vehicle counts and click <b>"Run PSO Optimization"</b> above
      </div>
      <div style="font-size:13px; color:#8b949e; margin-top:6px;">
        The AI will calculate optimal green-light timings for all 4 roads.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#484f58; font-size:12px; font-family:"Share Tech Mono",monospace;'>
PSO Smart Traffic Signal Optimizer &nbsp;|&nbsp; AI Course CCP Project &nbsp;|&nbsp;
Built with Python + Streamlit
</p>
""", unsafe_allow_html=True)
