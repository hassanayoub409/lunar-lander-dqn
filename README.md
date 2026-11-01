# Lunar Lander DQN

A small app about reinforcement learning, with everything implemented **from scratch in NumPy**:

1. **Lunar Lander agent.** A Deep Q-Network (DQN) learns, from trial and error only, to land a spacecraft between two flags. See it before and after training, its learning curve and test results.
2. **Mars rover: Q-values.** An interactive explainer. Change the rewards, the discount factor and the chance of moving the wrong way, and watch the optimal action in every state update live.

[Open the app](https://lunar-lander-dqn-409.streamlit.app) 

## Run the app

Requires Python 3.9+. The trained agent and replay GIFs are included, so the app needs no simulator.

```bash
git clone https://github.com/hassanayoub409/lunar-lander-dqn.git
cd lunar-lander-dqn
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the agent (about 3 minutes on one CPU core)

```bash
pip install -r requirements-train.txt
python train.py
```

This needs [Gymnasium](https://gymnasium.farama.org/) with Box2D. If installing Box2D fails, install SWIG first (macOS: `brew install swig`) and retry. Training stops once the average score over 100 episodes reaches 235, evaluates on 100 unseen episodes, and re-renders the GIFs.


## What's interesting about the implementation

- **No deep learning framework.** The Q-network, backpropagation, Adam optimizer, experience replay and target network are plain NumPy (`src/`). The hand-written gradients are checked against numerical gradients in `tests/`. The only extra dependency is Gymnasium, and only for training.
- **Double DQN targets and a Huber loss** for more stable learning than the basic algorithm.
- **Correct handling of time limits.** An episode cut off by the 1000-step limit is not treated as a terminal state, so the agent still bootstraps from the next state.
- **Honest evaluation.** Test episodes use random seeds never seen during training, with exploration switched off. The replay GIFs use fixed seeds that were not picked for how they turned out.
- **Exact Q-values for the Mars rover.** Solved by repeatedly applying the Bellman equation until convergence, and tested against the textbook example (values 100, 50, 25, 12.5, 20, 40).

## Project structure

```
app.py               # Streamlit app
train.py             # Train + evaluate the agent, render GIFs
src/dqn.py           # Q-network, replay buffer, DQN agent
src/rover.py         # Mars rover Bellman solver
src/optim.py         # Adam optimizer
models/              # Trained weights + training history
assets/              # Replay GIFs
tests/               # Gradient checks and unit tests (python -m pytest -q)
.streamlit/config.toml  # App colour theme
```

## Acknowledgments

The concepts behind this project (Q-learning, the Bellman equation, deep Q-networks with experience replay) were learned in the *Machine Learning Specialization* by DeepLearning.AI and Stanford Online on Coursera. The code in this repository was written independently and contains no course materials. The simulator is LunarLander from [Gymnasium](https://gymnasium.farama.org/) (Farama Foundation, MIT license; physics by Box2D). This project is not affiliated with or endorsed by Coursera, DeepLearning.AI, Stanford Online or the Farama Foundation.

## License

MIT, see [LICENSE](LICENSE).
