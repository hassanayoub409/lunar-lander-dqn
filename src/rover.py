"""The 'Mars rover' toy problem: a row of states, a reward at each end, actions left/right.

Computes the optimal state-action values Q*(s, a) by repeatedly applying the Bellman equation:

    Q(s, a) = step_reward + gamma * [ (1 - p) * V(intended next state) + p * V(other next state) ]
    V(s)    = max_a Q(s, a)

The two end states are terminal: their value is just their reward.
"""
import numpy as np

LEFT, RIGHT = 0, 1


def solve_rover(left_reward=100.0, right_reward=40.0, step_reward=0.0, gamma=0.5,
                misstep=0.0, n_states=6, max_iter=20_000, tol=1e-10) -> np.ndarray:
    """Return Q with shape (n_states, 2): column 0 = go left, column 1 = go right."""
    Q = np.zeros((n_states, 2))
    Q[0, :], Q[-1, :] = left_reward, right_reward
    for _ in range(max_iter):
        V = Q.max(axis=1)
        new = Q.copy()
        for s in range(1, n_states - 1):
            new[s, LEFT] = step_reward + gamma * ((1 - misstep) * V[s - 1] + misstep * V[s + 1])
            new[s, RIGHT] = step_reward + gamma * ((1 - misstep) * V[s + 1] + misstep * V[s - 1])
        if np.max(np.abs(new - Q)) < tol:
            return new
        Q = new
    return Q


def best_actions(Q, tol=1e-9):
    """'left', 'right', 'tie' for each state (terminal states -> 'end')."""
    out = ["end"]
    for s in range(1, len(Q) - 1):
        if abs(Q[s, LEFT] - Q[s, RIGHT]) < tol:
            out.append("tie")
        else:
            out.append("left" if Q[s, LEFT] > Q[s, RIGHT] else "right")
    return out + ["end"]
