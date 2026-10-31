"""Run from the repo root:  python -m pytest -q"""
import numpy as np

from src.dqn import DQNAgent, QNetwork, ReplayBuffer
from src.rover import best_actions, solve_rover


def test_qnetwork_gradients_match_numerical():
    rng = np.random.default_rng(0)
    net = QNetwork((5, 7, 6, 3), seed=1)
    x, w = rng.normal(size=(9, 5)), rng.normal(size=(9, 3))
    loss = lambda: float((net.forward(x) * w).sum())  # linear in the outputs, so dLoss/dOut = w
    _, acts = net.forward(x, keep_cache=True)
    grads = net.backward(w, acts)
    for name, arr in net.params.items():
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in arr.shape)
            old = arr[idx]
            arr[idx] = old + 1e-6
            up = loss()
            arr[idx] = old - 1e-6
            down = loss()
            arr[idx] = old
            assert np.isclose(grads[name][idx], (up - down) / 2e-6, rtol=1e-4, atol=1e-7)


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(5, 2)
    for k in range(12):
        buf.add(np.array([k, k]), k % 3, float(k), np.array([k + 1, k + 1]), k == 11)
    assert len(buf) == 5
    s, a, r, s2, d = buf.sample(4)
    assert s.shape == (4, 2) and set(r) <= {7.0, 8.0, 9.0, 10.0, 11.0}


def test_agent_learns_a_trivial_bandit():
    """One state, 2 actions: action 1 always pays +1, action 0 pays -1."""
    agent = DQNAgent(state_dim=1, n_actions=2, hidden=8, gamma=0.0, lr=0.01, batch_size=32, seed=0)
    for _ in range(300):
        a = int(agent.rng.integers(2))
        agent.buffer.add(np.array([1.0]), a, 1.0 if a == 1 else -1.0, np.array([1.0]), True)
    for _ in range(300):
        agent.learn()
    assert agent.online.act(np.array([1.0])) == 1


def test_rover_matches_the_textbook_example():
    Q = solve_rover(100, 40, 0, gamma=0.5, misstep=0)
    assert np.allclose(Q.max(axis=1), [100, 50, 25, 12.5, 20, 40])
    assert best_actions(Q) == ["end", "left", "left", "left", "right", "end"]


def test_rover_is_symmetric_when_every_move_is_a_coin_flip():
    Q = solve_rover(100, 40, 0, gamma=0.9, misstep=0.5)
    assert np.allclose(Q[1:-1, 0], Q[1:-1, 1])
