# Towards More Optimal Policies in Continuous Action Spaces

Learning in real-world domains often requires to deal with continuous state and action spaces and although reinforcement learning has been quite succesful with dealing with discrete state - action spaces and continuous state spaces, it has faced challenges extending up to continous action spaces. In this project, we look at possible improvements to the Deep Deterministic Policy Gradient (DDPG) algorithm which is state of the art for continuous action spaces. One of the recent algorithm called Twin Delayed Deep Deterministic policy gradients builds on DDPG and tries to improve it by reducing overestimation bias in value estimates. The algorithm employs a pair of critics taking the minimum between the two to restrict overestimation and delays policy updates to reduce per-update error. We propose a Prioritized experience replay based improvement to the algorithm, called Prioritized Twin Delayed Deep Deterministic olicy gradients (PTD3) and compare its performance to the vanilla TD3 and DDPG algorithm. Also, we test how the performance of TD3 and PTD3 changes with changes in different hyperparameter values. The study shows that PTD3 consistently outperforms TD3 on Mujoco Continuous control tasks, performing on an average 30\% better. PTD3 also demonstrates more stable training process and is less sensitive to hyperparameters.