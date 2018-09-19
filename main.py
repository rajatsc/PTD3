import numpy as np
import torch
import gym
import argparse
import os
import sys
import utils
import TD3
import per


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="TD3")					# Policy name
	parser.add_argument("--env_name", default="HalfCheetah-v1")			# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)	 	# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1000, type=float)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_false")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=4, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--train_time", default=2000, type=int)
	parser.add_argument("--prioritized", action="store_false")
	args = parser.parse_args()

	file_name = "%s_%s_%s_%s_%s_%s_%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed), str(args.expl_noise), str(args.discount), str(args.tau), str(args.policy_noise), str(args.noise_clip), str(args.policy_freq))
	print("---------------------------------------")
	print("Settings: %s" % (file_name))
	print("---------------------------------------")

	print("--------Creating directories----------")
	if not os.path.exists("./results"):
		print("Creating results directory")
		os.makedirs("./results")
	else:
		print("results already exists")
	if args.save_models and not os.path.exists("./pytorch_models"):
		print("Creating pytorch models directory")
		os.makedirs("./pytorch_models")
	else:
		print("pytorch models already exists")

	print("------Finding trained file--------")
	my_file_path="./results/" + file_name + '_COMPLETE' + ".npy"
	if os.path.exists(my_file_path):
		print("File already exists.....")
		print("Exiting Main")
		sys.exit(0)


	print("------Making Environment---------")
	env = gym.make(args.env_name)

	# Set seeds
	print("-------Setting Seeds-----------")
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	print("-------Get state and action dimension---------")
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	print('state_dim: ', state_dim)
	print('action_dim: ', action_dim)

	print("-------Get the value of max and min action -----------")
	max_action = int(env.action_space.high[0])
	min_action = int(env.action_space.low[0])
	print("max_action: ", max_action)
	print("min_action:", min_action)

	# Initialize policy
	print("-----------Initialize Algorithm--------------")
	print("Initializing"+ args.policy_name)
	if args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)
	elif args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)

	"""
	alpha=0.6
	memory_size=10000
	replay_buffer=proportional_replay.Experience(memory_size=memory_size, batch_size=args.batch_size, alpha=alpha)
	"""

	
	capacity=100000
	print("---------Initialize Replay Buffer-----------")
	if args.prioritized:
		replay_buffer = per.PriorExpReplay(capacity)
	else:	
		replay_buffer = utils.ReplayBuffer()
	
	# Evaluate untrained policy
	print("------Evaluating untrained policy---------")
	evaluations=[]
	evaluations.append([utils.evaluate_policy(policy, args.env_name), 0])



	#print("------Running time-------")

	"""
	while total_timesteps < args.max_timesteps:
		print("Timestep: ", total_timesteps+1)
		
		if done: 

			if total_timesteps != 0: 
				print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
				if args.policy_name == "TD3":
					policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
				else: 
					policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
			
			# Evaluate episode
			if timesteps_since_eval >= args.eval_freq:
				timesteps_since_eval %= args.eval_freq
				evaluations.append(evaluate_policy(policy))
				
				if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
				np.save("./results/%s" % (file_name), evaluations) 
			
			# Reset environment
			print("Resetting environment")
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
		
		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs))
			if args.expl_noise != 0: 
				action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

		# Perform action
		print('Performing actions')
		new_obs, reward, done, _ = env.step(action) 
		done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		episode_reward += reward


		# Store data in replay buffer
		replay_buffer.add((obs, new_obs, action, reward, done_bool))

		obs = new_obs

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1
		
	# Final evaluation
	print('---------Final Evaluation----------')
	evaluations.append(evaluate_policy(policy))
	if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
	np.save("./results/%s" % (file_name), evaluations)  


	"""

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	done = True 

	while (total_timesteps < args.max_timesteps):
		#print('Timestep: ', total_timesteps+1)		

		#Reset environment
		#print("Resetting environment")
		if done:

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		##---------- Evaluate and save evaluations --------------------##

		if timesteps_since_eval >= args.eval_freq:
			timesteps_since_eval %= args.eval_freq
			evaluations.append([utils.evaluate_policy(policy, args.env_name), total_timesteps])

				
			if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
			np.save("./results/%s" % (file_name), evaluations)


		## ----------- Select action either randomly or according to actor ------------##

		if total_timesteps < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs))
			if args.expl_noise != 0: 
				action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)



		new_obs, reward, done, info = env.step(action) 
		done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		episode_reward += reward

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1



		
		#print('Storing experience in replay buffer')
		# Store data in replay buffer
		if args.prioritized:
			replay_buffer.store_transition(obs, action, reward, new_obs, done_bool)
		else:
			replay_buffer.add((obs, new_obs, action, reward, done_bool))
		obs=new_obs
		#print(replay_buffer)
		

		#print('Total timesteps: ', total_timesteps)
		#print('Episode_timesteps: ', episode_timesteps)
		if (total_timesteps > args.train_time):
			#print('Hi I can begin train')
				
			if args.policy_name == "TD3":
				policy.train(replay_buffer, total_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq, args.prioritized)
			else: 
				policy.train(replay_buffer, total_timesteps, args.batch_size, args.discount, args.tau)

		

	# Final evaluation
	print('---------Final Evaluation----------')
	evaluations.append([utils.evaluate_policy(policy, args.env_name), total_timesteps])
	if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
	np.save("./results/%s" % (file_name), evaluations)

	#append 'COMPLETE' at the end of models and evaluation files

	old_file_path="./results/" + file_name + ".npy"
	os.rename(old_file_path, my_file_path)



				
		
