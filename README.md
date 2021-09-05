# Tetris
Perfecting Tetris using Deep Reinforcement Learning.



This study attempts to crack the game of Tetris using model-free Deep Reinforcement Learning
and compares the performance of value-based methods against policy-based methods. In
this project, we define perfecting the game of Tetris as clearing simultaneously as many lines
are possible. The main algorithms used are Deep Q-Networks, DQN, and Proximal Policy
optimizations, PPO. We experiment with different state representations of the game, the
effect of the discount rate and the importance of a good reward function. Most of the early
implementation and testing is done on multiple small size boards, instead of the normal 20x10,
to increase training efficiency and allow for parameter tuning. The policy-based methods are
able to perfect the game of Tetris using a smaller board clearing the maximum amount of
lines simultaneously and overall 1000 lines per episode, but fail to replicate their performance
on the big board, clearing less than 10 lines. The value-based methods create an endurance
behavior and clear lots of single lines to maximize game play time. This results in clearing
over 2000 lines per episode on the small board and over 10 lines on the big board. Overall,
this project showcases the difficulties and complications that model-free Deep Reinforcement
Learning has to face.
