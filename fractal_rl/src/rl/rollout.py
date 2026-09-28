"""GAE: bootstrap time-limit truncations, but do not bridge reset episodes."""
import numpy as np


def compute_gae(rewards, values, next_values, terminated, ended, gamma, gae_lambda):
    # Arrays [time, env]. Terminated suppresses bootstrapping, ended suppresses GAE carryover.
    advantage = np.zeros_like(rewards, dtype=np.float32)
    following = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        following = delta + gamma * gae_lambda * (1 - ended[t]) * following
        advantage[t] = following
    return advantage, advantage + values
