"""Speed reward integrates distance so maximizing return maximizes average speed."""
import numpy as np


def reward(distance, action, previous_action, dt, cfg):
    action = np.asarray(action)
    previous_action = np.asarray(previous_action)
    return float(cfg['speed_weight'] * distance - dt * (
        cfg['control_penalty'] * np.dot(action, action)
        + cfg['action_change_penalty'] * np.sum((action - previous_action)**2)))
