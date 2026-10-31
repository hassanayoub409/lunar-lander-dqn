"""Deep Q-Network (DQN) for discrete-action environments, in plain NumPy.

Ingredients: a Q-network, an experience-replay buffer, a slowly-moving target network
(soft updates), epsilon-greedy exploration, and the Double-DQN target. This module has
no dependency on gymnasium, so the app and the tests run without it.
"""
import numpy as np

from .optim import Adam


class QNetwork:
    """MLP state -> one Q-value per action, with hand-written backpropagation."""

    def __init__(self, sizes=(8, 64, 64, 4), seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.params = {}
        for k, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
            self.params[f"W{k}"] = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
            self.params[f"b{k}"] = np.zeros(n_out)
        self.n_layers = len(sizes) - 1

    def forward(self, x, keep_cache=False):
        activations, h = [x], x
        for k in range(1, self.n_layers + 1):
            h = h @ self.params[f"W{k}"] + self.params[f"b{k}"]
            if k < self.n_layers:
                h = np.maximum(h, 0.0)
            activations.append(h)
        return (h, activations) if keep_cache else h

    def backward(self, d_out, activations):
        """Gradients of a loss w.r.t. all parameters, given dLoss/dOutput."""
        grads, d = {}, d_out
        for k in range(self.n_layers, 0, -1):
            a_prev = activations[k - 1]
            grads[f"W{k}"] = a_prev.T @ d
            grads[f"b{k}"] = d.sum(axis=0)
            if k > 1:
                d = (d @ self.params[f"W{k}"].T) * (activations[k - 1] > 0)
        return grads

    def soft_update_from(self, other, tau):
        for k in self.params:
            self.params[k] *= 1.0 - tau
            self.params[k] += tau * other.params[k]

    def copy_from(self, other):
        for k in self.params:
            self.params[k][...] = other.params[k]

    def save(self, path):
        np.savez(path, sizes=np.array(self.sizes), **self.params)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        net = cls(tuple(int(s) for s in data["sizes"]))
        for k in net.params:
            net.params[k] = data[k]
        return net

    def act(self, state):
        """Greedy action for a single state."""
        return int(np.argmax(self.forward(state[None, :])[0]))


class ReplayBuffer:
    """Fixed-size ring buffer of (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity, state_dim, seed=0):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float64)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float64)
        self.d = np.zeros(capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, state_dim=8, n_actions=4, hidden=64, gamma=0.99, lr=1e-3, tau=1e-3,
                 batch_size=64, buffer_size=100_000, seed=0):
        sizes = (state_dim, hidden, hidden, n_actions)
        self.online, self.target = QNetwork(sizes, seed), QNetwork(sizes, seed)
        self.target.copy_from(self.online)
        self.opt = Adam(self.online.params, lr=lr)
        self.buffer = ReplayBuffer(buffer_size, state_dim, seed)
        self.gamma, self.tau, self.batch_size, self.n_actions = gamma, tau, batch_size, n_actions
        self.rng = np.random.default_rng(seed + 1)

    def select_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.online.act(state)

    def td_targets(self, r, s2, done):
        """Double-DQN: the online net picks the next action, the target net evaluates it."""
        best_next = np.argmax(self.online.forward(s2), axis=1)
        q_next = self.target.forward(s2)[np.arange(len(r)), best_next]
        return r + self.gamma * (1.0 - done) * q_next

    def learn(self):
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        y = self.td_targets(r, s2, done)
        q, acts = self.online.forward(s, keep_cache=True)
        rows = np.arange(len(a))
        err = q[rows, a] - y
        d_out = np.zeros_like(q)
        d_out[rows, a] = np.clip(err, -1.0, 1.0) / len(a)  # gradient of the Huber loss
        self.opt.step(self.online.backward(d_out, acts))
        self.target.soft_update_from(self.online, self.tau)
        return float(np.mean(np.where(np.abs(err) <= 1, 0.5 * err ** 2, np.abs(err) - 0.5)))
