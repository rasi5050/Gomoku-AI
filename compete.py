import itertools as it
from time import perf_counter
from argparse import ArgumentParser
import gomoku as gm
from policies import policy_classes

# @profile
def compete(board_size, win_size, policies, verbose=True):

    runtimes = {gm.MAX: 0, gm.MIN: 0}

    state = gm.GomokuState.blank(board_size, win_size)

    for turn in it.count():

        # print current state
        if verbose: print(state)

        # stop if game over
        if state.is_game_over(): break

        # get policy for current player
        player = state.current_player()
        policy = policies[player]

        # run policy to select an action and time it
        turn_start = perf_counter()
        action = policy(state.copy())
        turn_duration = perf_counter() - turn_start
        runtimes[player] += turn_duration

        # check action validity
        if action not in state.valid_actions():
            raise ValueError(f"{type(policy).__name__} policy returned invalid action")

        # perform action
        state = state.perform(action)

        if verbose: print(f"Turn {turn}: {type(policy).__name__} took {turn_duration} seconds")

    score = state.current_score()
    return score, runtimes


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-b", "--board-size", type=int, choices=list(range(3, 20)), default=15, help="The size of the board (3 to 19)")
    parser.add_argument("-w", "--win-size", type=int, choices=list(range(2, 20)), default=5, help="The number in a row to win (3 to 19)")
    parser.add_argument("-x", "--max-player", help="The classname for the max player policy (string)")
    parser.add_argument("-o", "--min-player", help="The classname for the min player policy (string)")
    args = parser.parse_args()

    if args.win_size > args.board_size:
        raise ValueError(f"Win size ({args.win_size}) should be no more than board size ({args.board_size})")

    if args.max_player is None: args.max_player = "Human"
    if args.min_player is None: args.min_player = "Human"

    all_policies = {cls.__name__: cls for cls in policy_classes}

    policies = {
        gm.MAX: all_policies[args.max_player](args.board_size, args.win_size),
        gm.MIN: all_policies[args.min_player](args.board_size, args.win_size),
    }

    score, runtimes = compete(args.board_size, args.win_size, policies)
    print(f"Final score: {score}")
    print(f"Runtimes: max={runtimes[gm.MAX]}, min={runtimes[gm.MIN]}")

    print(args)

