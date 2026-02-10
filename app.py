# ============================================================
# Wumpus World Simulator (Streamlit UI)
# ------------------------------------------------------------
# This file implements a visual and interactive user interface
# for the Wumpus World environment defined in wumpus_env.py.
#
# The environment logic (rules, rewards, percepts) is kept
# completely separate from the UI, following the standard
# Agent–Environment separation described in Russell & Norvig.
# ============================================================

import time
import random
from collections import defaultdict

# Data handling and plotting
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit for UI / interaction
import streamlit as st

# Wumpus World environment and agent definitions
from wumpus_env import WumpusWorldEnv, NaiveAgent, Action, Pos


# ============================================================
# Streamlit Page Configuration
# ============================================================

# Configure the Streamlit page layout
st.set_page_config(page_title="WumpusWorld Simulator", layout="wide")
st.title("WumpusWorld — Game")


# ============================================================
# Custom CSS Styling
# ------------------------------------------------------------
# This section defines custom CSS to make the interface
# more readable and game-like (HUD cards, badges, board).
# ============================================================

st.markdown("""
<style>
.hud {display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px;}
.card {
  padding:10px 14px; border:1px solid rgba(120,120,120,.35);
  border-radius:14px; background: rgba(255,255,255,.03);
  min-width: 150px;
}
.card .k {font-size:12px; opacity:0.7;}
.card .v {font-size:20px; font-weight:700; margin-top:2px;}
.badges {display:flex; gap:8px; flex-wrap:wrap; margin-top:6px; margin-bottom:6px;}
.badge {
  padding:4px 10px; border-radius:999px;
  border:1px solid rgba(120,120,120,.35);
  font-size:13px;
}
.boardWrap {margin-top:8px; margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HUD Rendering
# ------------------------------------------------------------
# Displays the current global state of the environment
# (steps, score, agent status, inventory, etc.)
# ============================================================

def render_hud(env):
    """
    Render a top HUD summarizing the current environment state.
    This information is NOT available to the agent — it is
    only for visualization/debugging.
    """
    status = "☠️ DEAD" if env.dead else ("🏁 TERMINAL" if env.terminated else "🟢 RUNNING")
    arrow = "🏹 Yes" if env.arrow_available else "— No"
    gold = "✨ Yes" if env.has_gold else "— No"
    w = "👾 Alive" if env.wumpus_alive else "💀 Dead"

    html = f"""
    <div class='hud'>
      <div class='card'><div class='k'>Status</div><div class='v'>{status}</div></div>
      <div class='card'><div class='k'>Steps</div><div class='v'>{env.steps}</div></div>
      <div class='card'><div class='k'>Score</div><div class='v'>{env.total_reward}</div></div>
      <div class='card'><div class='k'>Gold</div><div class='v'>{gold}</div></div>
      <div class='card'><div class='k'>Arrow</div><div class='v'>{arrow}</div></div>
      <div class='card'><div class='k'>Wumpus</div><div class='v'>{w}</div></div>
      <div class='card'><div class='k'>Pos / Dir</div><div class='v'>({env.agent_pos.x},{env.agent_pos.y}) {env.agent_dir.name}</div></div>
    </div>
    """
    return html


# ============================================================
# Percept Visualization
# ------------------------------------------------------------
# Shows which percepts are currently TRUE using icons.
# These correspond exactly to the percept tuple defined
# in the environment.
# ============================================================

def percept_badges(per):
    """
    Visual representation of the agent's percepts
    (stench, breeze, glitter, bump, scream).
    """
    icons = {
        "stench": ("stench", "🤢"),
        "breeze": ("breeze", "💨"),
        "glitter": ("glitter", "✨"),
        "bump": ("bump", "🧱"),
        "scream": ("scream", "😱"),
    }

    on = []
    for k, (name, icon) in icons.items():
        if getattr(per, k):
            on.append(f"<span class='badge'>{icon} {name}</span>")

    if not on:
        on.append("<span class='badge'>✅ no percepts</span>")

    return "<div class='badges'>" + "".join(on) + "</div>"


# ============================================================
# Board Rendering
# ------------------------------------------------------------
# Converts the debug board into an emoji-based grid.
# The agent only *sees* percepts; this board is for humans.
# ============================================================

def render_board_html(env, reveal=True):
    """
    Render the Wumpus World grid using emojis.
    If reveal=False, hazards are hidden (agent-view mode).
    """
    board = env.get_debug_board()

    # Direction indicator for the agent
    dir_emoji = {"N": "⬆️", "E": "➡️", "S": "⬇️", "W": "⬅️"}
    agent_icon = "🤖" + dir_emoji[env.agent_dir.name]

    def cell_emoji(cell_set):
        if "A" in cell_set:
            return agent_icon
        if not reveal:
            return "⬛"
        if "P" in cell_set:
            return "🕳️"
        if "W" in cell_set:
            return "👾"
        if "w" in cell_set:
            return "💀"
        if "G" in cell_set:
            return "✨"
        return "⬜"

    rows_html = []
    for y in range(env.height, 0, -1):
        cells = []
        for x in range(1, env.width + 1):
            p = Pos(x, y)
            emoji = cell_emoji(board[p])
            cells.append(
                f"<td style='text-align:center;font-size:28px;width:64px;height:64px;"
                f"border:1px solid rgba(120,120,120,.35); border-radius:12px;"
                f"background: rgba(255,255,255,.03);'>{emoji}</td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return "<table style='border-collapse:separate;border-spacing:6px;'>" + "".join(rows_html) + "</table>"


# ============================================================
# Logging Utility
# ------------------------------------------------------------
# Records each step of the simulation for later analysis,
# plotting, and CSV export.
# ============================================================

def log_step(env, action, percept):
    """
    Capture the full state of the environment after each action.
    """
    return dict(
        step=env.steps,
        action=action.name,
        reward=percept.reward,
        score=env.total_reward,
        stench=percept.stench,
        breeze=percept.breeze,
        glitter=percept.glitter,
        bump=percept.bump,
        scream=percept.scream,
        pos_x=env.agent_pos.x,
        pos_y=env.agent_pos.y,
        dir=env.agent_dir.name,
        has_gold=env.has_gold,
        wumpus_alive=env.wumpus_alive,
        arrow_available=env.arrow_available,
        terminal=env.terminated,
        dead=env.dead,
    )


# ============================================================
# Environment Factory
# ------------------------------------------------------------
# Creates a fresh environment instance with given parameters.
# ============================================================

def make_env(width, height, allow_climb, pit_prob, seed):
    return WumpusWorldEnv(width, height, allow_climb, pit_prob, seed=seed)


# ============================================================
# Sidebar Controls (Experiment Parameters)
# ============================================================

seed = st.sidebar.number_input("Seed", value=7, step=1)
width = st.sidebar.slider("Width", 2, 10, 4)
height = st.sidebar.slider("Height", 2, 10, 4)
pit_prob = st.sidebar.slider("pitProb", 0.0, 0.6, 0.2, 0.05)
allow_climb = st.sidebar.checkbox("allowClimbWithoutGold", value=True)
delay = st.sidebar.slider("Delay (s)", 0.0, 1.0, 0.15, 0.05)
max_steps = st.sidebar.slider("Max steps", 1, 5000, 200, 1)

view_mode = st.sidebar.selectbox("Board view", ["Reality (debug)", "Agent view (hidden)"])
reveal = (view_mode == "Reality (debug)")


# ============================================================
# Session State Initialization
# ------------------------------------------------------------
# Streamlit re-runs the script on each interaction, so we
# store environment state explicitly in session_state.
# ============================================================

if "env" not in st.session_state:
    st.session_state.env = make_env(width, height, allow_climb, pit_prob, int(seed))
    st.session_state.agent = NaiveAgent(seed=int(seed) + 1)
    st.session_state.log = []
    st.session_state.running = False
    st.session_state.params = (seed, width, height, pit_prob, allow_climb)


def reset_episode():
    """Reset environment, agent, and logs."""
    st.session_state.env = make_env(width, height, allow_climb, pit_prob, int(seed))
    st.session_state.agent = NaiveAgent(seed=int(seed) + 1)
    st.session_state.log = []
    st.session_state.running = False


# Auto-reset if experiment parameters change
params_now = (seed, width, height, pit_prob, allow_climb)
if st.session_state.params != params_now:
    st.session_state.params = params_now
    reset_episode()
    st.rerun()


# ============================================================
# Control Buttons (Step / Play / Pause)
# ============================================================

colA, colB, colC, colD = st.columns(4)

if colA.button("Reset"):
    reset_episode()
    st.rerun()

if colB.button("Step (Agent)"):
    env = st.session_state.env
    if not env.is_terminal():
        per = env.get_percept(0)
        action = st.session_state.agent.choose_action(per)
        percept = env.step(action)
        st.session_state.log.append(log_step(env, action, percept))
    st.rerun()

if colC.button("Step (Random)"):
    env = st.session_state.env
    if not env.is_terminal():
        action = random.choice(list(Action))
        percept = env.step(action)
        st.session_state.log.append(log_step(env, action, percept))
    st.rerun()

with colD:
    run_col, stop_col = st.columns(2)
    if run_col.button("Play"):
        st.session_state.running = True
        st.rerun()
    if stop_col.button("Pause"):
        st.session_state.running = False
        st.rerun()


# ============================================================
# Automatic Runner (Animation Loop)
# ============================================================

env = st.session_state.env
if st.session_state.running:
    if (not env.is_terminal()) and (env.steps < max_steps):
        per = env.get_percept(0)
        action = st.session_state.agent.choose_action(per)
        percept = env.step(action)
        st.session_state.log.append(log_step(env, action, percept))
        time.sleep(delay)
        st.rerun()
    else:
        st.session_state.running = False
        st.rerun()


# ============================================================
# Layout: Game (Left) / Telemetry (Right)
# ============================================================

env = st.session_state.env
per = env.get_percept(0)

left, right = st.columns([1.15, 1])

with left:
    st.markdown(render_hud(env), unsafe_allow_html=True)
    st.markdown(percept_badges(per), unsafe_allow_html=True)
    st.markdown("<div class='boardWrap'></div>", unsafe_allow_html=True)
    st.markdown(render_board_html(env, reveal=reveal), unsafe_allow_html=True)

    if env.is_terminal():
        if env.dead:
            st.error("Episode ended: Agent is DEAD.")
        else:
            st.success("Episode ended: Agent CLIMBED out.")
    elif env.steps >= max_steps:
        st.warning("Episode stopped: reached Max steps.")

    with st.expander("Debug details"):
        st.write(dict(
            pos=(env.agent_pos.x, env.agent_pos.y),
            dir=env.agent_dir.name,
            has_gold=env.has_gold,
            wumpus_alive=env.wumpus_alive,
            arrow_available=env.arrow_available,
            steps=env.steps,
            score=env.total_reward,
            terminal=env.terminated,
            dead=env.dead
        ))
        st.write("Last percept tuple:", per.as_tuple())

with right:
    st.subheader("Telemetry")
    log_df = pd.DataFrame(st.session_state.log)

    tab1, tab2, tab3 = st.tabs(["Recent", "Charts", "Export"])

    with tab1:
        if not log_df.empty:
            st.dataframe(log_df.tail(20), use_container_width=True)
        else:
            st.info("No log yet. Click Step or Play to generate data.")

    with tab2:
        if not log_df.empty:
            fig1 = plt.figure()
            plt.plot(log_df["step"], log_df["score"])
            plt.xlabel("Step")
            plt.ylabel("Score")
            st.pyplot(fig1)

            fig2 = plt.figure()
            plt.plot(log_df["step"], log_df["reward"])
            plt.xlabel("Step")
            plt.ylabel("Reward")
            st.pyplot(fig2)
        else:
            st.info("No data to chart yet.")

    with tab3:
        if not log_df.empty:
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download log CSV", csv, "wumpus_log.csv", "text/csv")
        else:
            st.info("Nothing to export yet.")
