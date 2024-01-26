"""
Do not change this file, it will be replaced by the instructor's copy
"""
import itertools as it
import random
import numpy as np
from scipy.signal import correlate
import gomoku as gm

# helper function to get minimal path length to a game over state
# @profile
def turn_bound(state):

    is_max = state.is_max_turn()
    fewest_moves = state.board[gm.EMPTY].sum() # moves to a tie game

    # use correlations to extract possible routes to a non-tie game
    corr = state.corr
    min_routes = (corr[:,gm.EMPTY] + corr[:,gm.MIN] == state.win_size)
    max_routes = (corr[:,gm.EMPTY] + corr[:,gm.MAX] == state.win_size)
    # also get the number of turns in each route until game over
    min_turns = 2*corr[:,gm.EMPTY] - (0 if is_max else 1)
    max_turns = 2*corr[:,gm.EMPTY] - (1 if is_max else 0)

    # check if there is a shorter path to a game-over state
    if min_routes.any():
        moves_to_win = min_turns.flatten()[min_routes.flatten()].min()
        fewest_moves = min(fewest_moves, moves_to_win)
    if max_routes.any():
        moves_to_win = max_turns.flatten()[max_routes.flatten()].min()
        fewest_moves = min(fewest_moves, moves_to_win)

    # return the shortest path found to a game-over state
    return fewest_moves

# helper to find empty position in pth win pattern starting from (r,c)
def find_empty(state, p, r, c):
    if p == 0: # horizontal
        return r, c + state.board[gm.EMPTY, r, c:c+state.win_size].argmax()
    if p == 1: # vertical
        return r + state.board[gm.EMPTY, r:r+state.win_size, c].argmax(), c
    if p == 2: # diagonal
        rng = np.arange(state.win_size)
        offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
        return r + offset, c + offset
    if p == 3: # antidiagonal
        rng = np.arange(state.win_size)
        offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
        return r - offset, c + offset
    # None indicates no empty found
    return None

# fast look-aheads to short-circuit the minimax search when possible
def look_ahead(state):

    # if current player has a win pattern with all their marks except one empty, they can win next turn
    player = state.current_player()
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum() # no +1 since win comes after turn

    # check if current player is one move away to a win
    corr = state.corr
    idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size-1))
    if idx.shape[0] > 0:
        # find empty position they can fill to win, it is an optimal action
        p, r, c = idx[0]
        action = find_empty(state, p, r, c)
        return sign * magnitude, action

    # else, if opponent has at least two such moves with different empty positions, they can win in two turns
    opponent = gm.MIN if state.is_max_turn() else gm.MAX
    loss_empties = set() # make sure the 2+ empty positions are distinct
    idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size-1))
    for p, r, c in idx:
        pos = find_empty(state, p, r, c)
        loss_empties.add(pos)        
        if len(loss_empties) > 1: # just found a second empty
            score = -sign * (magnitude - 1) # opponent wins an extra turn later
            return score, pos # block one of their wins with next action even if futile

    # return 0 to signify no conclusive look-aheads
    return 0, None

# recursive minimax search with additional pruning
# @profile
def minimax(state, max_depth, alpha=-np.inf, beta=np.inf):
    # print("called inside AI, max_depth=", max_depth)
    # check fast look-ahead before trying minimax
    score, action = look_ahead(state)
    if score != 0: return score, action

    # check for game over base case with no valid actions
    if state.is_game_over():
        return state.current_score(), None

    # have to try minimax, prepare the valid actions
    # should be at least one valid action if this code is reached
    actions = state.valid_actions()

    # prioritize actions near non-empties but break ties randomly
    rank = -state.corr[:, 1:].sum(axis=(0,1)) - np.random.rand(*state.board.shape[1:])
    rank = rank[state.board[gm.EMPTY] > 0] # only empty positions are valid actions
    scrambler = np.argsort(rank)

    
    # check for max depth base case
    if max_depth == 0:
        return state.current_score(), actions[scrambler[0]]

    # custom pruning: stop search if no path from this state wins within max_depth turns
    if turn_bound(state) > max_depth: return 0, actions[scrambler[0]]

    # alpha-beta pruning
    best_action = None
    if state.is_max_turn():
        bound = -np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            utility, _ = minimax(child, max_depth-1, alpha, beta)

            if utility > bound: bound, best_action = utility, action
            if bound >= beta: break
            alpha = max(alpha, bound)

    else:
        bound = +np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            utility, _ = minimax(child, max_depth-1, alpha, beta)

            if utility < bound: bound, best_action = utility, action
            if bound <= alpha: break
            beta = min(beta, bound)

    return bound, best_action

# Policy wrapper
class Minimax:
    def __init__(self, board_size, win_size, max_depth=4):
        self.max_depth = max_depth

    def __call__(self, state):
        _, action = minimax(state, self.max_depth)
        
        return action

if __name__ == "__main__":

    # unit tests for look-ahead function

    state = gm.GomokuState.blank(5, 3)
    state = state.play_seq([(0,0), (0,1), (1,1), (1,2)])
    score, action = look_ahead(state)
    assert score == 1 + 5**2 - 5
    assert action == (2,2)

    state = gm.GomokuState.blank(5, 3)
    state = state.play_seq([(4,1), (4,2), (3,2), (3,3)])
    score, action = look_ahead(state)
    assert score == 1 + 5**2 - 5
    assert action == (2,3)

    print("no fails")

