import random

class Random:
    def __init__(self, board_size, win_size): pass

    def __call__(self, state):
        valid_actions = state.valid_actions()
        action = random.choice(valid_actions)
        return action

