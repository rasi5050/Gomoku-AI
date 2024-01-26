"""
Do not change this file, it will be replaced by the instructor's copy
"""
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import gomoku as gm
from policies import *
from compete import compete

# number of games used to calculate performance statistics
num_reps = 30
# num_reps = 100

# standard Gomoku parameters used for evaluation
BOARD_SIZE = 15
WIN_SIZE = 5

# AI policies for each player
policies = {
    # # random vs baseline
    # gm.MAX: Random(BOARD_SIZE, WIN_SIZE),
    # gm.MIN: Minimax(BOARD_SIZE, WIN_SIZE),

    # # baseline vs baseline
    # gm.MAX: Minimax(BOARD_SIZE, WIN_SIZE),
    # gm.MIN: Minimax(BOARD_SIZE, WIN_SIZE),

    # baseline vs submission
    gm.MAX: Minimax(BOARD_SIZE, WIN_SIZE),
    gm.MIN: Submission(BOARD_SIZE, WIN_SIZE),
}
# human-readable names of each policy
names = {player: type(policies[player]).__name__ for player in (gm.MIN, gm.MAX)}

# change to False to visualize results without rerunning the games
if True:

    all_scores = []
    all_runtimes = {gm.MAX: [], gm.MIN: []}
    
    print("Starting competition, may take a while...")
    for rep in range(num_reps):
    
        score, runtimes = compete(BOARD_SIZE, WIN_SIZE, policies, verbose=False)
        print(f"rep {rep} of {num_reps}: score={score}, min time ({names[gm.MIN]}) = {runtimes[gm.MIN]}, max time ({names[gm.MAX]}) = {runtimes[gm.MAX]}")

        all_scores.append(score)
        for player in (gm.MIN, gm.MAX):        
            all_runtimes[player].append(runtimes[player])
    
    with open("perf.pkl","wb") as f: pk.dump((all_scores, all_runtimes), f)

with open("perf.pkl","rb") as f: (all_scores, all_runtimes) = pk.load(f)

pt.subplot(1,3,1)
pt.hist(all_scores, ec='k')
pt.plot([np.mean(all_scores)], [0], 'ro')
pt.ylabel("Frequency")
pt.xlabel("Final score")
pt.subplot(1,3,2)
pt.hist(all_runtimes[gm.MIN], ec='k')
pt.xlabel(f"{type(policies[gm.MIN]).__name__} run time (min player)")
pt.subplot(1,3,3)
pt.hist(all_runtimes[gm.MAX], ec='k')
pt.xlabel(f"{type(policies[gm.MAX]).__name__} run time (max player)")
pt.tight_layout()

pt.show()

