# Gomoku-AI

#### Gomoku-AI is an AI developed for fulfilling the requirements for CIS667(Introduction to Artificial Intelligence) Final Project Fall 2023. This respository contains the boilerplate code and our implementation of CNN + minimax heuristic function.

## Team Members
[Sanchit Relan](https://github.com/srelan46) [Niranjan Balasubramani](https://github.com/NiranjanBalasubramani) [Nilesh Bhoi](https://github.com/nilesh507) [Rasi](https://github.com/rasi5050)

## Introduction

In this project we have combined a custom neural network with a custom minimax algorithm to play Gomoku on a 15x15
board. The neural network learns game states and adapts from game scenarios, enhancing strategy along the way. If
neural network gives invalid action, we use the minimax algorithm with a custom evaluation function to strategically
minimize losses and maximize gains to produce optimal moves. This dual approach not only makes the AI highly skilled in
Gomoku but also makes it adaptable to various playing styles, displaying a state-of-the-art AI and game theory.

## Key Features of the Custom Minimax

The custom heuristic evaluation function dynamically allocates the score and weighs the board state. It scans the board
in various directions (horizontal, vertical, diagonal, and anti-diagonal) and aggregates scores based on consecutive
actions by the player on the board in continuous direction and blocks the opponent two moves before it is about to win.
Scores have more weight when the game is close to completion.

The MIN player chooses an aggressive strategy and gets assigned higher scores for potential winning sequences
(get_line_score() function), promoting a more proactive and offensive gameplay. Due to the weighted evaluation favoring
the MIN player, the MAX player often resorts to a defensive approach, focusing on blocking rather than creating its own
threats. The aggressive approach led to early victories for the MIN player, with many games concluding rapidly in favor of
the Submission. The MAX player's reactive behavior and defensive strategy, primarily focused on countering the threats
posed by the MIN player, often led to missed opportunities for offense. Due to the chosen strategy, the MIN player wins
early for most games but also loses early sometimes. In a few cases, the performance of the MIN player outweighs that
of the MAX player and is quick to finish a game.

## Key Features of the Custom Neural Network
Input Data: In the project 2880 input and output game states are chosen from the Gomo-cup competition,2022.
Neural Network Architecture: We have used a sequential convolutional neural network (CNN) with six convolutional
layers. The architecture first increases and then decreases the depth of the network, starting with 64, escalating to 256
and going back to 64(64->128->256->128->64), Activation function used in this layer is RELU. The final layer is a single
1x1 convolutional layer which is then followed up by a reshape layer with sigmoid activation function, for binary
classification of the predicted state. In the project, we have used the Adam optimizer in our neural network, which
produces better results than SGD in our scenario. Also, epoch size of 20 was optimal in this scenario.

## Training Results 

As mentioned below in the figure, we are getting an accuracy of around 84% and loss of 0.042.

<img width="703" alt="Screenshot 2024-01-26 at 14 04 39" src="https://github.com/rasi5050/Gomoku-AI/assets/12760472/a6d8ad57-2a97-453b-ba74-1def7085bb60">

<img width="689" alt="Screenshot 2024-01-26 at 14 04 48" src="https://github.com/rasi5050/Gomoku-AI/assets/12760472/bf62257d-ece4-49f1-9414-ad65f4e84034">

## Results:

On average, we are winning 75% of the games and above with a deviation of ±𝟓% consistently.
![1_26_24 12-46PM](https://github.com/rasi5050/Gomoku-AI/assets/12760472/177943ab-8db2-42fe-bb97-a6e3b1bd6078)

![sanchit new + niranjan new](https://github.com/rasi5050/Gomoku-AI/assets/12760472/65425387-8401-4b77-a5a9-0b3c0183a9a2)


## Discussion:

One of the approaches we followed in our custom neural network was to reduce the number of layers and increase the
number of epochs. This resulted in a slow increase in accuracy as the epochs increased. The learning curve was slower
and took more compute time to train. The trained model was also overfitting the data, resulting in less wins compared to
our current approach.
The other approach that we tried involved achieving a balance between - accessing the model and custom minmax
function to calculate the action manually. When the depth limit is reached, the model predicts the utility to arrive at
optimal action. However, it introduced a significant overhead while calculating utilities of all children at each step. In
future scope, this problem can be addressed by performing n-fold cross-validation.



## Code Structure(These were made available in the boilerplate code)

• gomoku.py: This module contains the implementation of the Gomoku game, including methods to
list valid actions for every state, perform an action in a given state, and calculate the score in a game
state.
• compete.py: This module runs a full game between two players. Each player can be controlled by a
human or various automated policies.
• performance.py: This module runs several games between your AI and the baseline policy. The
plots at the end visualize the final scores and run times of each AI. It saves the results in a file
named perf.pkl.
• policies/: The modules in this sub-directory include various policies that can be selected for each
player.
o human.py: This policy is human-controlled
o random.py: This policy chooses actions uniformly at random
o minimax.py: This policy chooses actions using an augmented Minimax Search
o submission.py: This policy will use our AI implementation


## run the game

You can run a game between two policies by running compete.py on the command line with options to
specify board size, win size, and each player’s policy. 

For example, the command

`python3 compete.py -b 15 -w 5 -x Human -o Minimax`

will run a game on a 15x15 board with 5 in a row to win, where you control the max player, and the min
player selects actions randomly. For each policy you must write the exact name of the policy class (casesensitive). The available policies are Human, Random, Minimax, and Submission (which is yours). Press
Ctrl-C, or a similar command in your operating system, to interrupt the script at any time and terminate
early before the game is over.

the command

`python3 performance.py `

will run 30 games against base line AI (this was used to compare performace in the above screenshots)

## Teams contribution 

team has contributed to create the CNN+minimax heuristic function which can be see at 
1. policies/submission.py, which includes minimax code and model invocation.
2. policies/CNN+model directory, which includes CNN creation, datasets and models(best model we obtained was named "model_2800_20epoch) 

