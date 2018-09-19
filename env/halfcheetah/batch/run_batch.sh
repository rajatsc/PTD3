#!/bin/bash

ENV='HalfCheetah-v2'
START_T=10000
MAX_TIMESTEPS=500000
TRAIN_T=2000


s_array=(1 2 3)    #seed
en_array=(0.1)   #exploratory noise  
t_array=(0.005)    #tau
pn_array=(0.2)   #policy noise
nc_array=(0.5)  #noise clip
pf_array=(4)   #policy_freq
batch_array=(500)

for s in "${s_array[@]}"
do
  for en in "${en_array[@]}"
  do
    for t in "${t_array[@]}"
    do
      for pn in "${pn_array[@]}"
      do
        for nc in "${nc_array[@]}"
        do
          for pf in "${pf_array[@]}"
          do
            for batch in "${batch_array[@]}"
            do
              python3.5 ../../../main.py \
              --policy_name "TD3" \
              --env_name $ENV \
              --seed $s \
              --start_timesteps $START_T \
              --eval_freq 5000 \
              --max_timesteps $MAX_TIMESTEPS \
              --expl_noise $en \
              --batch_size $batch \
              --discount 0.99 \
              --tau $t \
              --policy_noise $pn \
              --noise_clip $nc \
              --policy_freq $pf \
              --train_time $TRAIN_T
            done
          done
        done
      done
    done 
  done
done
