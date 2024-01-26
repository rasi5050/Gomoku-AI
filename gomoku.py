"""
Do not change this file, it will be replaced by the instructor's copy
"""
import sys
import itertools as it
import numpy as np

# enum for grid cell contents
EMPTY = 0
MIN = 1
MAX = 2
SYMBOLS = np.array(['.','o','x'])

class GomokuState:
    def __init__(self, board, win_size, corr=None):
        """
        Initialize a state with the given board and win size

        corr is the correlation of each win pattern with the board
        it is maintained by blank() and perform() and should not be modified
        corr invariants:
        corr[0,c,row,col] = board[c,row, col:col+ws].sum() # horizontal
        corr[1,c,row,col] = board[c,row:row+ws, col].sum() # vertical
        corr[2,c,row,col] = board[c,range(row,row+ws), range(col,col+ws)].sum() # diagonal
        corr[3,c,row,col] = board[c,rnage(row,row-ws), range(col,col+ws)].sum() # antidiagonal

        """
        self.board = board
        self.win_size = win_size
        self.corr = corr

        # cached return values after first call to their respective getters
        self.player = None
        self.score = None
        self.over = None
        self.actions = None

    def __str__(self):
        """
        Return string representation of this state
        """
        symbol_index = self.board.argmax(axis=0)
        return "\n".join("".join(row) for row in SYMBOLS[symbol_index])

    def current_player(self):
        """
        Get current player in this state
        """
        if self.player is None:
            equal_symbols = (self.board[MAX].sum() == self.board[MIN].sum())
            self.player = [MIN, MAX][int(equal_symbols)]
        return self.player

    def is_max_turn(self):
        """
        Return True iff the current player is the max player
        """
        return (self.current_player() == MAX)

    def current_score(self):
        """
        Return the current score in this state
        """
        if self.score is None:

            self.score = 0
            magnitude = 1 + self.board[EMPTY].sum() # larger score for earlier win
            player_signs = [(MIN, -1), (MAX, +1)]

            # check for maximal correlation with win patterns
            for player, sign in player_signs:
                if (self.corr[:, player] == self.win_size).any():
                    self.score = sign * magnitude
                    break

        return self.score

    def is_game_over(self):
        """
        Return True iff the game is over this state
        """
        if self.over is None:
            self.over = (not self.board[EMPTY].any()) or (self.current_score() != 0)
        return self.over

    def valid_actions(self):
        """
        Return a tuple of valid actions in this state
        """
        if self.actions is None:
            if self.is_game_over(): self.actions = ()
            else: self.actions = tuple(zip(*np.nonzero(self.board[EMPTY])))
        return self.actions

    # @profile
    def perform(self, action):
        """
        Perform the given action in this state and return a new state
        """
        player = self.current_player()
        row, col = action

        # convert selected board position from empty to current player
        new_board = self.board.copy()
        new_board[EMPTY, row, col] = 0
        new_board[player, row, col] = 1

        # update the win pattern correlations to reflect the change in the board
        corr = self.corr.copy()
        rng = np.arange(self.win_size)
        # corr[0,c,row,col] = board[c,row, col:col+ws].sum() # horizontal win
        idx = col - rng
        idx = idx[(0 <= idx) & (idx + self.win_size <= self.board.shape[-1])]
        corr[0, EMPTY, row, idx] -= 1
        corr[0, player, row, idx] += 1
        # corr[1,c,row,col] = board[c,row:row+ws, col].sum() # vertical win
        idx = row - rng
        idx = idx[(0 <= idx) & (idx + self.win_size <= self.board.shape[-2])]
        corr[1, EMPTY, idx, col] -= 1
        corr[1, player, idx, col] += 1
        # corr[2,c,row,col] = board[c,row:row+ws, col:col+ws].sum() # diagonal win
        idx = np.array([[row],[col]]) - rng
        idx = idx[:, (0 <= idx).all(axis=0) & (idx[0] + self.win_size <= self.board.shape[-2]) & (idx[1] + self.win_size <= self.board.shape[-1])]
        corr[2, EMPTY, idx[0], idx[1]] -= 1
        corr[2, player, idx[0], idx[1]] += 1
        # corr[3,c,row,col] = board[c,row:row-ws, col:col+ws].sum() # anti-diagonal win
        idx = np.stack([row + rng, col - rng])
        idx = idx[:, (self.win_size-1 <= idx[0]) & (idx[0] < self.board.shape[-1])]
        idx = idx[:, (0 <= idx[1]) & (idx[1] + self.win_size <= self.board.shape[-2])]
        corr[3, EMPTY, idx[0], idx[1]] -= 1
        corr[3, player, idx[0], idx[1]] += 1

        # return modified game state
        return GomokuState(new_board, self.win_size, corr)

    def copy(self):
        """
        Return a copy of this state
        """
        corr = None if self.corr is None else self.corr.copy()
        state = GomokuState(self.board, self.win_size, corr)
        state.player = self.player
        state.score = self.score
        state.over = self.over
        state.actions = self.actions
        return state

    def blank(board_size, win_size):
        """
        Static class method that returns the initial game state with a blank board
        """
        # empty board
        board = np.zeros((3, board_size, board_size))
        board[EMPTY,:,:] = 1

        # initial correlations
        # corr[0,c,row,col] = board[c,row, col:col+ws].sum() # horizontal
        # corr[1,c,row,col] = board[c,row:row+ws, col].sum() # vertical
        # corr[2,c,row,col] = board[c,row:row+ws, col:col+ws].sum() # diagonal
        # corr[3,c,row,col] = board[c,row:row-ws, col:col+ws].sum() # antidiagonal
        corr = np.zeros((4,) + board.shape, dtype=int)
        corr[0,EMPTY,:,:-win_size+1] = win_size
        corr[1,EMPTY,:-win_size+1,:] = win_size
        corr[2,EMPTY,:-win_size+1,:-win_size+1] = win_size
        corr[3,EMPTY,win_size-1:,:-win_size+1] = win_size

        state = GomokuState(board, win_size, corr)
        return state

    # helper to generate and test intermediate game states
    def play_seq(self, actions, midgame=True):
        state = self
        for action in actions:
            if midgame: assert not state.is_game_over()
            state = state.perform(action)
        return state


if __name__ == "__main__":

    # unit tests

    state = GomokuState.blank(5, 3)

    assert not state.is_game_over()
    assert state.current_score() == 0
    assert state.current_player() == MAX

    state = state.play_seq([(0,0), (0,1), (1,0), (1,1), (2,0)])
    assert state.is_game_over()
    assert state.current_score() == 1 + 5**2 - 5
    assert state.current_player() == MIN

    state = GomokuState.blank(5, 3)
    state = state.play_seq([(0,0), (1,0), (0,1), (1,1), (0,2)])

    assert state.is_game_over()
    assert state.current_score() == 1 + 5**2 - 5
    assert state.current_player() == MIN

    state = GomokuState.blank(5, 3)
    state = state.play_seq([(0,0), (1,0), (1,1), (2,1), (2,2)])

    assert state.is_game_over()
    assert state.current_score() == 1 + 5**2 - 5
    assert state.current_player() == MIN

    state = GomokuState.blank(5, 3)
    state = state.play_seq([(0,4), (1,4), (1,3), (2,3), (2,2)])

    assert state.is_game_over()
    assert state.current_score() == 1 + 5**2 - 5
    assert state.current_player() == MIN

    state = GomokuState.blank(5, 3)
    state = state.play_seq([(0,2), (1,2), (1,1), (2,1), (2,0)])

    assert state.is_game_over()
    assert state.current_score() == 1 + 5**2 - 5
    assert state.current_player() == MIN

    state = GomokuState.blank(5, 3)
    state = state.play_seq([(2,2), (3,2), (3,1), (4,1), (4,0)])

    assert state.is_game_over()
    assert state.current_score() == 1 + 5**2 - 5
    assert state.current_player() == MIN

    state = GomokuState.blank(5, 3)
    state = state.play_seq([(2,2), (3,2), (3,1), (4,1), (0,0), (2,3)])

    assert state.is_game_over()
    assert state.current_score() == - (1 + 5**2 - 6)
    assert state.current_player() == MAX

    print("no fails")

