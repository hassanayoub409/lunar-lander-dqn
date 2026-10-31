"""Train a DQN agent on LunarLander, evaluate it, and render replay GIFs.

    python train.py                  # full run (needs: pip install -r requirements-train.txt)
    python train.py --episodes 5     # quick smoke test

Outputs: models/lander_dqn.npz, models/history.json, assets/*.gif
"""
import argparse
import json
import time

import gymnasium as gym
import numpy as np
from PIL import Image

from src.dqn import DQNAgent, QNetwork
from src.paths import ASSETS_DIR, MODELS_DIR

ENV_ID = "LunarLander-v3"
SOLVED = 200  # score at which an episode counts as a successful landing


def train(max_episodes, target_avg, eps_decay, seed, max_minutes, log=print):
    env = gym.make(ENV_ID)
    agent = DQNAgent(state_dim=8, n_actions=4, seed=seed)
    scores, epsilons, best_avg, best_params = [], [], -np.inf, None
    eps, start = 1.0, time.time()
    for ep in range(max_episodes):
        state, _ = env.reset(seed=seed if ep == 0 else None)
        total = 0.0
        for t in range(1000):
            action = agent.select_action(state, eps)
            nxt, reward, terminated, truncated, _ = env.step(action)
            # Only a real termination zeroes the future; a time-limit cut-off does not.
            agent.buffer.add(state, action, reward, nxt, terminated)
            if t % 4 == 0 and len(agent.buffer) >= 1000:
                agent.learn()
            state, total = nxt, total + reward
            if terminated or truncated:
                break
        eps = max(0.01, eps * eps_decay)
        scores.append(total)
        epsilons.append(eps)
        avg = float(np.mean(scores[-100:]))
        if ep >= 99 and avg > best_avg:
            best_avg = avg
            best_params = {k: v.copy() for k, v in agent.online.params.items()}
        if ep % 25 == 0 or ep == max_episodes - 1:
            log(f"episode {ep:4d} | score {total:7.1f} | avg100 {avg:7.1f} | eps {eps:.3f} "
                f"| {(time.time() - start) / 60:.1f} min")
        if avg >= target_avg and ep >= 99:
            log(f"Reached average {avg:.1f} over 100 episodes at episode {ep}.")
            break
        if (time.time() - start) / 60 > max_minutes:
            log("Time limit reached.")
            break
    env.close()
    if best_params is not None:
        for k, v in best_params.items():
            agent.online.params[k][...] = v
    else:  # fewer than 100 episodes were played (smoke test)
        best_avg = float(np.mean(scores[-100:]))
    return agent.online, scores, epsilons, best_avg


def evaluate(net, n_episodes=100, seed0=10_000):
    """Greedy policy on fresh seeds the agent never trained on."""
    env = gym.make(ENV_ID)
    scores = []
    for k in range(n_episodes):
        state, _ = env.reset(seed=seed0 + k)
        total, done = 0.0, False
        while not done:
            state, r, term, trunc, _ = env.step(net.act(state))
            total, done = total + r, term or trunc
        scores.append(total)
    env.close()
    return scores


def render_gif(policy, seed, path, frame_skip=2, max_frames=400, size=(480, 320)):
    """Run one episode and save it as a GIF. Returns (score, n_steps)."""
    env = gym.make(ENV_ID, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames, total, steps, done = [], 0.0, 0, False
    while not done:
        state, r, term, trunc, _ = env.step(policy(state))
        total, steps, done = total + r, steps + 1, term or trunc
        if steps % frame_skip == 0 and len(frames) < max_frames:
            frames.append(Image.fromarray(env.render()).resize(size))
    env.close()
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
    return total, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--target", type=float, default=235.0, help="stop when 100-episode average reaches this")
    ap.add_argument("--eps-decay", type=float, default=0.992)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=60)
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    net, scores, epsilons, best_avg = train(args.episodes, args.target, args.eps_decay,
                                            args.seed, args.max_minutes)
    net.save(MODELS_DIR / "lander_dqn.npz")

    print("Evaluating on 100 unseen episodes...")
    test = evaluate(net)
    rng = np.random.default_rng(0)
    gif_info = {}
    for seed in (1, 2, 3):  # fixed seeds, not picked by outcome
        score, steps = render_gif(net.act, seed, ASSETS_DIR / f"trained_{seed}.gif")
        gif_info[f"trained_{seed}"] = {"score": round(score, 1), "steps": steps}
    score, steps = render_gif(lambda s: int(rng.integers(4)), 1, ASSETS_DIR / "random.gif")
    gif_info["random"] = {"score": round(score, 1), "steps": steps}

    history = {
        "scores": [round(s, 1) for s in scores],
        "epsilon": [round(e, 4) for e in epsilons],
        "best_avg100": round(best_avg, 1),
        "test": {
            "episodes": len(test),
            "mean": round(float(np.mean(test)), 1),
            "std": round(float(np.std(test)), 1),
            "success_rate": round(float(np.mean(np.array(test) >= SOLVED)), 3),
            "min": round(float(np.min(test)), 1),
            "max": round(float(np.max(test)), 1),
        },
        "gifs": gif_info,
        "settings": {"env": ENV_ID, "eps_decay": args.eps_decay, "seed": args.seed,
                     "gamma": 0.99, "lr": 1e-3, "tau": 1e-3, "batch_size": 64,
                     "buffer_size": 100_000, "hidden": [64, 64], "update_every": 4},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (MODELS_DIR / "history.json").write_text(json.dumps(history))
    print(json.dumps({k: history[k] for k in ("best_avg100", "test", "gifs", "minutes")}, indent=2))


if __name__ == "__main__":
    main()
