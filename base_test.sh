#!/bin/bash

ENV=HalfCheetah-v2

rm -rf results

python3.5 ./main.py \
--policy_name "TD3" \
--env_name $ENV \
--seed 1 \
--start_timesteps 10000 \
--eval_freq 5000 \
--max_timesteps 500000 \
--expl_noise 0.1 \
--batch_size 100 \
--discount 0.99 \
--tau 0.005 \
--policy_noise 0.2 \
--noise_clip 0.5 \
--policy_freq 4 \
--train_time 2000 
