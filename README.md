# Tetris
Perfecting Tetris using Deep Reinforcement Learning.



<p align="justify">This study attempts to crack the game of Tetris using model-free Deep Reinforcement Learning
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


<p align="justify"> If you want to replicate the results it is suggested that you use the 8x6 boards since the big Tetris board needs quite the training (3 weeeks for the DQN agent). The code above is set the train the agents using the glipmse version. If you wish to train the agent on any other state represenations (4feat, 12feat, board) please make sure that the NN hyperpaters are changed appropriately and change the mode of the game to version you wish. For any further questions please dont hesitate to email me on antreaskafkalias7@gmail.com  </p>

Lets visualize the agents:

PPO on the 8x6 board!

![ppo](https://user-images.githubusercontent.com/72248364/132142545-f40431ba-c89a-4d6d-ba9c-0d1df024b055.gif =250x250)


DQN on the 8x6 board!

![dqn](https://user-images.githubusercontent.com/72248364/132142550-b42f6332-f706-4c90-b1d3-1612bd6a105a.gif)

PPO on the 20x10 board!


DQN on the 20x10 board!






