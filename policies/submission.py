"""
Do not change this file, it will be replaced by the instructor's copy
"""
import itertools as it
import random
import numpy as np
from scipy.signal import correlate
import gomoku as gm
from tensorflow.keras.models import load_model

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

# The get_line_score function assigns higher scores for potential winning sequences for the 
# MIN player, promoting a more proactive and offensive gameplay.
def get_line_score(count, blocks, max_player):
    MAX_SCORE = 100000000
    if blocks == 2 and count < 5: # Both ends blocked
        return 0

    if count >= 5: # which means the current player is winning
        return MAX_SCORE

    if count == 4:
        if blocks == 0:
            return MAX_SCORE // 4 if not max_player else 500000        

    if count == 3:
        if blocks == 0:
            return 5000 if not max_player else 200
        else:
            return 10 if not max_player else 5

    if count == 2:
        return 8 if not max_player and blocks == 0 else 4
    
    if count == 1:
        return 1

    return 0

def evaluate_board_for_player(board, player):
    score = 0
    # horizontal, vertical, diagonal, anti-diagonal
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for direction in directions:
        score += evaluate_direction(board, player, direction)

    return score

def evaluate_direction(board, player, direction):
    dx, dy = direction
    score = 0
    board_size = board.shape[1]
    
    for x in range(board_size):
        for y in range(board_size):
            # 3 Dimensional numpy array: 
            # first dimension: player, remaining two: coordinates
            # Checks whether the cell at coordinates (x,y) is occupied by the player
            if board[player, x, y] == 1: 
                count, blocks = 1, 2  # Start with both sides blocked

                # Check if left/top side of the line is open
                if (x - dx >= 0 and y - dy >= 0) and (x - dx < board_size and y - dy < board_size) and board[player, x - dx, y - dy] == 0:
                    blocks -= 1
                # Check if right/bottom side of the line is open
                if (x + dx < board_size and y + dy < board_size) and (x + dx >= 0 and y + dy >= 0) and board[player, x + dx, y + dy] == 0:
                    blocks -= 1

                # Count consecutive marks in the direction
                nx, ny = x + dx, y + dy
                while 0 <= nx < board_size and 0 <= ny < board_size and board[player, nx, ny] == 1:
                    count += 1
                    nx += dx
                    ny += dy

                score += get_line_score(count, blocks, player == gm.MAX)

    return score

def dynamic_heuristic_weighting(state, early_game_weight, late_game_weight):
    
    total_cells = state.board.shape[1] * state.board.shape[2]
    empty_cells = state.board[gm.EMPTY].sum()

    # Early game is defined when more than half of the board is empty
    if empty_cells > total_cells / 2:
        return early_game_weight
    else:
        return late_game_weight

# Custom heuristic evaluation function that dynamically allocates the score and weighs the board state. 
# Scans the board in various directions (horizontal, vertical, diagonal, and anti-diagonal) and 
# aggregates scores based on potential threats and opportunities.
def heuristic_evaluation(state):
    max_player = gm.MAX
    min_player = gm.MIN

    weight = dynamic_heuristic_weighting(state, early_game_weight=1, late_game_weight=2)

    return (evaluate_board_for_player(state.board, max_player) * weight) - \
        (evaluate_board_for_player(state.board, min_player) * weight)


# recursive minimax search with additional pruning
# @profile
def enhanced_minimax(state, max_depth, alpha=-np.inf, beta=np.inf):

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
    
    # check for max depth base case, running heuristic for one depth above
    if max_depth < 2:
        # print("max_depth_2")
        return heuristic_evaluation(state), actions[scrambler[0]]        

    # custom pruning: stop search if no path from this state wins within max_depth turns
    if turn_bound(state) > max_depth: 
        # If the shortest path to a winning state (as determined by turn_bound(state)) is longer than 
        # the remaining depth of my search (max_depth), then don't bother searching deeper; 
        # just evaluate the current state heuristically.
        # print("turn_bound called")
        return heuristic_evaluation(state), actions[scrambler[0]]        

    # alpha-beta pruning
    best_action = None
    if state.is_max_turn():
        bound = -np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            # based on the score returned from the minimax algorithm, pruning happens and the respective best action is chosen
            utility, _ = enhanced_minimax(child, max_depth-1, alpha, beta)

            if utility > bound: bound, best_action = utility, action
            if bound >= beta: break
            alpha = max(alpha, bound)

    else:
        bound = +np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            utility, _ = enhanced_minimax(child, max_depth-1, alpha, beta)

            if utility < bound: bound, best_action = utility, action
            if bound <= alpha: break
            beta = min(beta, bound)

    return bound, best_action

# Policy wrapper
class Submission:
    def __init__(self, board_size, win_size, max_depth=4):
        self.max_depth = max_depth
        self.model = load_model('./policies/model_2800_20epoch')

    def __call__(self, state):


        input = np.zeros((15,15), dtype=int)

        # Set positions occupied by the min player to -1
        # print(type(state))
        # print(state)
        input[state.board[1] == 1] = -1

        # Set positions occupied by the max player to +1
        input[state.board[2] == 1] = 1

        input = np.array(input, np.float32).reshape((-1, 15, 15, 1))
        
        # print(input.shape)
        output = self.model.predict(input)
        output = output.reshape((15, 15))
        output_y, output_x = np.unravel_index(np.argmax(output), output.shape)

        # output_y, output_x=output_x, output_y

        if (output_y, output_x) not in state.valid_actions():
            # print("reached invalid state check")
            _, action = enhanced_minimax(state, self.max_depth)
            return action
        return (output_y, output_x)





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

