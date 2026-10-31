"""Streamlit app: Reinforcement Learning Lab (Lunar Lander DQN + Mars rover Q-values).

Run with:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from src.paths import ASSETS_DIR, MODELS_DIR
from src.rover import best_actions, solve_rover

GREEN, GOLD, AMBER, RED = "#4F7D32", "#FFD16A", "#F5B52E", "#D9361E"

st.set_page_config(page_title="Reinforcement Learning Lab", page_icon="🚀", layout="wide")

history_file = MODELS_DIR / "history.json"
if not history_file.exists():
    st.error("No trained agent found. Run `python train.py` first (see the README).")
    st.stop()
h = json.loads(history_file.read_text())

st.title("Lunar Lander Deep Q-learning and Mars rover Q-values")
st.caption("Deep Q-learning and Q-values, implemented from scratch in NumPy.")

tab_lander, tab_rover, tab_how = st.tabs(
    ["Lunar Lander agent", "Mars rover: Q-values", "How it works"])

# ---------------------------------------------------------------- Lunar Lander
with tab_lander:
    st.markdown(
        "An agent learns by trial and error to land a spacecraft between the two flags. It sees 8 numbers "
        "(position, velocity, angle, leg contact) and picks one of 4 actions each moment: do nothing, or fire the "
        "left, main or right engine. There are no examples to copy, only a score."
    )
    t = h["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average test score", f"{t['mean']:.0f}", help=f"Over {t['episodes']} unseen episodes, std {t['std']:.0f}")
    c2.metric("Successful landings", f"{t['success_rate']:.0%}", help="Share of test episodes scoring 200 or more")
    c3.metric("Training episodes", f"{len(h['scores'])}")
    c4.metric("Training time", f"{h['minutes']:.0f} min", help="One CPU core, including evaluation and GIF rendering")

    st.subheader("Before and after training")
    left, right = st.columns(2)
    left.image(str(ASSETS_DIR / "random.gif"),
               caption=f"Untrained (random actions): score {h['gifs']['random']['score']:.0f}")
    right.image(str(ASSETS_DIR / "trained_1.gif"),
                caption=f"Trained agent: score {h['gifs']['trained_1']['score']:.0f}")

    st.subheader("Learning curve")
    scores = pd.Series(h["scores"], name="score")
    curve = pd.DataFrame({
        "Episode score": scores,
        "Average of last 100": scores.rolling(100, min_periods=1).mean(),
        "Success threshold (200)": 200.0,
    })
    curve.index.name = "Episode"
    st.line_chart(curve, color=[AMBER, GREEN, RED])
    st.caption("Early on the agent mostly crashes (very negative scores). As exploration is reduced and the "
               "network improves, scores climb past the 200 mark.")

    st.subheader("More test landings")
    st.caption("Fixed random seeds (2 and 3), not hand-picked for how they turned out.")
    a, b = st.columns(2)
    for col, key in ((a, "trained_2"), (b, "trained_3")):
        info = h["gifs"][key]
        col.image(str(ASSETS_DIR / f"{key}.gif"), caption=f"Score {info['score']:.0f} ({info['steps']} steps)")
    st.markdown(
        f"Across {t['episodes']} test episodes the agent scored between {t['min']:.0f} and {t['max']:.0f}. "
        "In the lower-scoring episodes it usually still touches down between the flags, but keeps firing its engines "
        "instead of switching them off, burning fuel until the time limit. Fixing that is a good next improvement."
    )

# ---------------------------------------------------------------- Mars rover
ARROWS = {"left": "←", "right": "→", "tie": "↔"}


def card(background, text, border, inner):
    return (f"<div style='flex:1;min-width:110px;background:{background};color:{text};border:4px solid {border};"
            f"border-radius:12px;padding:12px 6px;text-align:center;line-height:1.5'>{inner}</div>")


with tab_rover:
    st.markdown(
        "A rover sits in one of 6 states. It can move left or right. The two end states are terminal and pay a "
        "reward. **Q(s, a)** is the total (discounted) reward the rover can expect if it takes action *a* in state *s* "
        "and then acts optimally. Move the sliders and watch the best action in each state change."
    )
    s1, s2, s3, s4, s5 = st.columns(5)
    left_r = s1.slider("Left end reward", -50, 200, 100)
    right_r = s2.slider("Right end reward", -50, 200, 40)
    step_r = s3.slider("Reward per step", -5.0, 5.0, 0.0, step=0.5)
    gamma = s4.slider("Discount factor γ", 0.0, 0.99, 0.5, step=0.01,
                      help="Low γ: only nearby rewards matter. High γ: far-away rewards matter too.")
    miss = s5.slider("Chance of moving the wrong way", 0.0, 0.5, 0.0, step=0.05)

    Q = solve_rover(left_r, right_r, step_r, gamma, miss)
    policy = best_actions(Q)

    cards = []
    for s in range(len(Q)):
        if policy[s] == "end":
            reward = left_r if s == 0 else right_r
            bg, fg = (GREEN, "#FFFFFF") if reward > 0 else (RED, "#FFFFFF") if reward < 0 else (AMBER, "#222222")
            cards.append(card(bg, fg, bg, f"<div style='font-size:0.8em'>State {s + 1} (end)</div>"
                                          f"<div style='font-size:1.6em;font-weight:700'>{reward:g}</div>"
                                          f"<div style='font-size:0.8em'>terminal reward</div>"))
        else:
            ql, qr = Q[s]
            cards.append(card(GOLD, "#222222", GREEN,
                              f"<div style='font-size:0.8em'>State {s + 1}</div>"
                              f"<div style='font-size:2em;font-weight:700'>{ARROWS[policy[s]]}</div>"
                              f"<div style='font-size:0.85em'>Q(←) = {ql:.1f}<br>Q(→) = {qr:.1f}</div>"))
    st.markdown(f"<div style='display:flex;gap:10px;margin:12px 0'>{''.join(cards)}</div>", unsafe_allow_html=True)

    goes_left = [str(s + 1) for s in range(1, len(Q) - 1) if policy[s] == "left"]
    goes_right = [str(s + 1) for s in range(1, len(Q) - 1) if policy[s] == "right"]
    summary = []
    if goes_left:
        summary.append(f"goes **left** from state(s) {', '.join(goes_left)}")
    if goes_right:
        summary.append(f"goes **right** from state(s) {', '.join(goes_right)}")
    st.markdown("The optimal rover " + (" and ".join(summary) if summary else "is indifferent everywhere") + ".")
    with st.expander("Things to try"):
        st.markdown(
            "- Raise **γ** towards 0.99: the far-away reward of 100 matters more, so the rover walks further to reach it.\n"
            "- Lower **γ** towards 0: the rover only cares about what is right next to it.\n"
            "- Make **Reward per step** negative (e.g. -2): every move costs something, so the rover prefers the closer exit.\n"
            "- Add a **chance of moving the wrong way**: values shrink, and the best choice can flip.\n"
            "- Set both end rewards equal and the chance of a wrong move to 0.5: every action becomes equally good (↔)."
        )

# ---------------------------------------------------------------- How it works
with tab_how:
    s = h["settings"]
    st.markdown(
        f"""
**The agent (Deep Q-Network).** A small neural network (8 inputs, two hidden layers of {s['hidden'][0]} units, 4 outputs)
estimates Q(state, action) for each of the 4 actions. The agent mostly picks the action with the highest estimate, but
with probability ε it explores a random one. ε starts at 1 and shrinks every episode.

**Learning from memory.** Every step is stored in a *replay buffer* of {s['buffer_size']:,} experiences. Every
{s['update_every']} steps the network trains on a random batch of {s['batch_size']} of them, which breaks the strong
correlation between consecutive moments. The training target is
`reward + γ · max Q(next state)`, computed with a slowly-updated *target network* so the target does not chase its own tail.

**Implementation notes.**
- Everything (network, backpropagation, Adam, replay buffer, Mars rover solver) is plain NumPy, with the
  gradients checked numerically in `tests/`. The only extra library is Gymnasium for the simulator, and only when training.
- Uses **Double DQN** targets (the online network chooses the next action, the target network scores it), a Huber loss,
  and treats time-limit cut-offs differently from real crashes or landings.
- Evaluated on 100 episodes with random seeds that were never used in training, with exploration switched off.
- Hyperparameters: γ = {s['gamma']}, learning rate {s['lr']}, soft target update τ = {s['tau']}, ε decay {s['eps_decay']} per episode.

**Limitations.** One training run with one random seed: results vary between runs. Lunar Lander is a classic benchmark
and says little about real-world robotics.
"""
    )

st.divider()
st.caption(
    "Simulator: LunarLander from Gymnasium (Farama Foundation, MIT license; physics by Box2D). "
    "The concepts were learned in the Machine Learning Specialization (DeepLearning.AI / Stanford Online on Coursera); "
    "this code is an independent re-implementation and contains no course materials."
)
